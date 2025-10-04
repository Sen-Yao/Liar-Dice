import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import LiarDiceEnv, Guess, Challenge
from agents.basic_agent import BasicRuleAgent, ProbabilisticBasicAgent
from rl_specialized.utils.state_encoder import create_state_encoder

# 为了保持实现简单，将自博弈相关的对手与对手池也放在本文件中
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None  # 若未安装，在加载策略型对手时会报错提示


class LiarDiceSingleAgentEnv(gym.Env):
    """将 PettingZoo 的多智能体 AEC 环境包装为 SB3 可训练的单智能体 Gym 环境。

    设计：
    - RL 控制 player_0，其余玩家使用 BasicRuleAgent 对手
    - 使用专用动作空间（min_count = n+1），动作空间与 env.action_to_object 一一对应
    - 观测为 Dict：
        { 'obs': Box(state_vector), 'action_mask': MultiBinary(action_dim) }
    - 每次 step() 内部会推进直到再次轮到 RL 或回合结束
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players: int = 2, dice_per_player: int = 5, render_mode: Optional[str] = None):
        super().__init__()
        assert num_players >= 2
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.render_mode = render_mode

        # 使用专用动作空间
        self._min_count = num_players + 1
        self._total_dice = num_players * dice_per_player
        counts_per_mode = max(0, self._total_dice - self._min_count + 1)
        self._action_dim = 1 + (2 * counts_per_mode * 6)

        # 状态编码器
        self._encoder = create_state_encoder(num_players=num_players, dice_per_player=dice_per_player)
        obs_dim = self._encoder.get_feature_size()

        # Gym spaces
        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "action_mask": gym.spaces.MultiBinary(self._action_dim),
        })
        self.action_space = gym.spaces.Discrete(self._action_dim)

        # 内部环境 & 对手
        self._env: Optional[LiarDiceEnv] = None
        self._opponents: Dict[str, BasicRuleAgent] = {}
        self._rl_agent_id = "player_0"
        self._pending_reset_reward: float = 0.0

    # --- Helpers ---
    def _build_obs(self) -> Dict[str, np.ndarray]:
        assert self._env is not None
        obs_dict = self._env.observe(self._rl_agent_id)
        state_vec = self._encoder.encode_observation(obs_dict)
        mask = self._env.get_action_mask(obs_dict)
        return {"obs": state_vec.astype(np.float32), "action_mask": mask.astype(np.int8)}

    def _step_opponents_until_rl_turn_or_done(self) -> Tuple[float, bool, bool]:
        """推进对手回合，直到轮到 RL 或回合结束。返回 (reward, terminated, truncated)"""
        total_reward = 0.0
        terminated = False
        truncated = False
        assert self._env is not None

        while self._env.agents and self._env.agent_selection != self._rl_agent_id:
            ag_id = self._env.agent_selection
            opp = self._opponents.get(ag_id)
            obs = self._env.observe(ag_id)
            action = opp.get_action(obs)
            self._env.step(action)
            total_reward += float(self._env.rewards.get(self._rl_agent_id, 0))
            terminated = all(self._env.terminations.get(a, False) for a in self._env.agents)
            truncated = all(self._env.truncations.get(a, False) for a in self._env.agents)
            if terminated or truncated:
                break
        return total_reward, terminated, truncated

    # --- Gym API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._env = LiarDiceEnv(
            num_players=self.num_players,
            dice_per_player=self.dice_per_player,
            render_mode=self.render_mode,
            use_specialized_action_space=True,
        )
        self._env.reset(seed=seed)

        # 初始化对手（除 RL）
        self._opponents = {}
        for i in range(self.num_players):
            agent_id = f"player_{i}"
            if agent_id != self._rl_agent_id:
                self._opponents[agent_id] = BasicRuleAgent(agent_id=agent_id, num_players=self.num_players)

        # 可能一开始就不是 RL 的回合，推进到 RL 或回合结束
        self._pending_reset_reward = 0.0
        first_seed = seed
        attempts = 0
        max_attempts = 64

        while True:
            reward, terminated, truncated = self._step_opponents_until_rl_turn_or_done()
            self._pending_reset_reward += reward

            if not (terminated or truncated):
                if self._env.agent_selection == self._rl_agent_id:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Opponent stepping did not reach RL agent turn;"
                        f" attempts={attempts}, agent_selection={self._env.agent_selection}"
                    )
                continue

            attempts += 1
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Unable to reach RL agent turn after repeated resets;"
                    f" attempts={attempts}, max_attempts={max_attempts},"
                    f" agent_selection={self._env.agent_selection},"
                    f" term_all={all(self._env.terminations.values())},"
                    f" trunc_all={all(self._env.truncations.values())}"
                )

            self._env.reset(seed=first_seed)
            first_seed = None
            # 清空上一局奖励，避免跨局累计
            self._pending_reset_reward = 0.0

        obs = self._build_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self._env is not None
        assert self._env.agent_selection == self._rl_agent_id, "Not RL agent's turn"

        # RL 动作：离散ID -> 动作对象
        action_obj = self._env.action_to_object(action)
        self._env.step(action_obj)

        reward_rl = self._pending_reset_reward + float(self._env.rewards.get(self._rl_agent_id, 0))
        self._pending_reset_reward = 0.0
        terminated = all(self._env.terminations.get(a, False) for a in self._env.agents)
        truncated = all(self._env.truncations.get(a, False) for a in self._env.agents)

        if not (terminated or truncated):
            # 推进对手
            r2, term2, trunc2 = self._step_opponents_until_rl_turn_or_done()
            reward_rl += r2
            terminated = term2 or terminated
            truncated = trunc2 or truncated

        if terminated or truncated:
            obs = self.observation_space.sample()
            # 返回一个占位观测，不会继续使用
        else:
            obs = self._build_obs()

        info = {}
        return obs, reward_rl, terminated, truncated, info

    def render(self):
        if self._env is not None:
            self._env.render()

    def close(self):
        if self._env is not None:
            self._env = None


# ---------------------------- 自博弈：对手与对手池 ----------------------------

class BasicRuleOpponent:
    """可参数化的规则对手（独立于 BasicRuleAgent，便于设置起手面值/阈值）"""

    def __init__(self, num_players: int, start_face: int = 4, challenge_offset: int = 3):
        self.num_players = num_players
        self.start_face = int(np.clip(start_face, 2, 6))
        self.challenge_threshold = self.num_players + int(challenge_offset)
        self.challenge_offset = int(challenge_offset)

    def get_description(self) -> str:
        return f"基础规则(起手{self.start_face},阈值+{self.challenge_offset})"

    def get_action(self, observation: Dict) -> Any:
        # 首轮：飞模式，count = n+1，face = 自定义起手面值
        if observation["last_guess"] is None:
            return Guess(mode='飞', count=self.num_players + 1, face=self.start_face)

        last_guess: Guess = observation["last_guess"]

        # 超过阈值则挑战
        if last_guess.count > self.challenge_threshold:
            return Challenge()

        # 否则"小步抬价"
        current_mode = last_guess.mode
        current_count = last_guess.count
        current_face = last_guess.face

        if current_face < 6:
            # 斋且面为1时，面值升到2
            if current_mode == '斋' and current_face == 1:
                return Guess(mode=current_mode, count=current_count, face=2)
            return Guess(mode=current_mode, count=current_count, face=current_face + 1)
        else:
            # 面到6后：数量+1，面值重置为最小合法
            new_face = 1 if current_mode == '斋' else 2
            return Guess(mode=current_mode, count=current_count + 1, face=new_face)


class PolicyOpponent:
    """使用已保存的 PPO 策略作为对手（推断在CPU上进行以节省显存）"""

    def __init__(self, model_path: str, device: str = "cpu"):
        if PPO is None:
            raise RuntimeError("stable-baselines3 未安装，无法加载策略对手")
        self.model = PPO.load(model_path, device=device)
        self.model_path = model_path

    def get_description(self) -> str:
        import os
        filename = os.path.basename(self.model_path)
        return f"策略快照({filename})"

    def predict_action_id(self, obs_dict: Dict) -> int:
        # 形状校验：在状态编码维度变更（如加入历史）后，旧快照将不兼容
        try:
            model_obs_space = getattr(self.model, 'observation_space', None)
            if isinstance(model_obs_space, gym.spaces.Dict):
                trained_dim = int(model_obs_space["obs"].shape[0])
                current_dim = int(obs_dict["obs"].shape[-1])
                if current_dim != trained_dim:
                    dim_diff = current_dim - trained_dim
                    # 推断可能的原因
                    if dim_diff == 12:
                        reason = "状态编码已扩展历史特征（新27维 vs 旧15维，2人游戏）"
                    elif dim_diff == -12:
                        reason = "状态编码已移除历史特征（或使用了旧快照）"
                    elif abs(dim_diff) <= 5:
                        reason = f"玩家数量可能变化（维度差异: {dim_diff}）"
                    else:
                        reason = f"编码方案或配置变化（维度差异: {dim_diff}）"

                    raise RuntimeError(
                        f"不兼容的策略快照: {reason}. "
                        f"快照维度={trained_dim}, 当前维度={current_dim}. "
                        f"请移除旧快照或使用兼容的状态编码重新训练。"
                    )
        except RuntimeError:
            raise
        except Exception:
            # 若检查失败不影响正常预测，仅作软失败
            pass

        action, _ = self.model.predict(obs_dict, deterministic=True)
        return int(action)


class OpponentPool:
    """对手池：混合规则对手与策略快照对手，按 rule_ratio 进行加权采样"""

    def __init__(self, num_players: int, rule_ratio: float = 1.0):
        self.num_players = num_players
        self.rule_ratio = float(np.clip(rule_ratio, 0.0, 1.0))
        # 规则类对手（含 BasicRuleOpponent 与 ProbabilisticRuleOpponent）
        self.rules: List[Any] = []
        self.policies: List[PolicyOpponent] = []
        # 统计信息
        self.usage_stats = {'basic_rule': 0, 'prob_rule': 0, 'policy': 0}
        self.last_sampled_opponents = []

    def add_rule(self, start_face: int = 4, challenge_offset: int = 3):
        self.rules.append(BasicRuleOpponent(self.num_players, start_face, challenge_offset))

    def add_prob_rule(self, theta_challenge: float = 0.25, target_raise: float = 0.60, max_extra_raise: int = 2):
        self.rules.append(ProbabilisticRuleOpponent(self.num_players, theta_challenge, target_raise, max_extra_raise))

    def add_policy(self, model_path: str, device: str = "cpu"):
        self.policies.append(PolicyOpponent(model_path, device=device))

    def set_rule_ratio(self, ratio: float):
        self.rule_ratio = float(np.clip(ratio, 0.0, 1.0))

    def sample(self) -> Any:
        if len(self.policies) == 0 or np.random.rand() < self.rule_ratio:
            # 规则对手
            if not self.rules:
                # 若池为空，回退到默认规则对手
                opponent = BasicRuleOpponent(self.num_players)
                self.usage_stats['basic_rule'] += 1
                return opponent
            opponent = np.random.choice(self.rules)
            # 统计类型
            if isinstance(opponent, BasicRuleOpponent):
                self.usage_stats['basic_rule'] += 1
            elif isinstance(opponent, ProbabilisticRuleOpponent):
                self.usage_stats['prob_rule'] += 1
            return opponent
        # 策略对手
        opponent = np.random.choice(self.policies)
        self.usage_stats['policy'] += 1
        return opponent

    def get_usage_stats(self) -> Dict[str, float]:
        """获取对手使用统计数据（百分比）"""
        total = sum(self.usage_stats.values())
        if total == 0:
            return {'basic_rule': 0.0, 'prob_rule': 0.0, 'policy': 0.0}
        return {k: (v / total) * 100 for k, v in self.usage_stats.items()}

    def get_pool_summary(self) -> str:
        """获取对手池概要信息"""
        basic_count = sum(1 for op in self.rules if isinstance(op, BasicRuleOpponent))
        prob_count = sum(1 for op in self.rules if isinstance(op, ProbabilisticRuleOpponent))
        policy_count = len(self.policies)
        return f"基础规则:{basic_count} | 概率规则:{prob_count} | 策略快照:{policy_count}"

    def reset_stats(self):
        """重置统计信息"""
        self.usage_stats = {'basic_rule': 0, 'prob_rule': 0, 'policy': 0}


class ProbabilisticRuleOpponent:
    """概率型规则对手：封装 ProbabilisticBasicAgent 以便用于对手池"""

    def __init__(self, num_players: int, theta_challenge: float = 0.25, target_raise: float = 0.60, max_extra_raise: int = 2):
        self.theta_challenge = theta_challenge
        self.target_raise = target_raise
        self.max_extra_raise = max_extra_raise
        self.agent = ProbabilisticBasicAgent(
            agent_id="opp",
            num_players=num_players,
            theta_challenge=theta_challenge,
            target_raise=target_raise,
            max_extra_raise=max_extra_raise,
        )

    def get_description(self) -> str:
        return f"概率规则(挑战{self.theta_challenge:.2f},加注{self.target_raise:.2f})"

    def get_action(self, observation: Dict) -> Any:
        return self.agent.get_action(observation)


class LiarDiceSelfPlayEnv(gym.Env):
    """自博弈单智能体 Gym 环境：
    - RL 控制 player_0，其余座位从对手池采样
    - 观测仍为 Dict(obs, action_mask)
    - 动作掩码与动作ID映射与专用动作空间一致
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        pool: OpponentPool,
        dice_per_player: int = 5,
        render_mode: Optional[str] = None,
        dense_shaping: bool = True,
        shaping_beta: float = 0.05,
        shaping_gamma: float = 0.99,
        # 训练期行为塑形（鼓励更长对局）：
        # - 过早挑战的软惩罚（在叫点数较少时，RL 选择 Challenge 会被小幅扣分）
        # - 合法加注的小额奖励（每次 RL 选择 Guess 会给小额 shaping 奖励）
        early_challenge_min_raises: int = 2,
        early_challenge_penalty: float = 0.2,
        guess_step_bonus: float = 0.02,
        show_opponent_info: bool = False,
    ):
        super().__init__()
        assert pool.num_players >= 2
        self.pool = pool
        self.num_players = pool.num_players
        self.dice_per_player = dice_per_player
        self.render_mode = render_mode
        # 奖励潜在塑形设置（默认开启）
        self.dense_shaping = dense_shaping
        self.shaping_beta = float(shaping_beta)
        self.shaping_gamma = float(shaping_gamma)
        # 行为塑形参数
        self.early_challenge_min_raises = int(max(0, early_challenge_min_raises))
        self.early_challenge_penalty = float(max(0.0, early_challenge_penalty))
        self.guess_step_bonus = float(max(0.0, guess_step_bonus))
        self._show_opponent_info = show_opponent_info

        # 专用动作空间尺寸（与 LiarDiceSingleAgentEnv 相同计算）
        self._min_count = self.num_players + 1
        self._total_dice = self.num_players * dice_per_player
        counts_per_mode = max(0, self._total_dice - self._min_count + 1)
        self._action_dim = 1 + (2 * counts_per_mode * 6)

        self._encoder = create_state_encoder(num_players=self.num_players, dice_per_player=dice_per_player)
        obs_dim = self._encoder.get_feature_size()

        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "action_mask": gym.spaces.MultiBinary(self._action_dim),
        })
        self.action_space = gym.spaces.Discrete(self._action_dim)

        self._env: Optional[LiarDiceEnv] = None
        self._rl_agent_id = "player_0"
        self._seat_opponents: Dict[str, Any] = {}
        # 用于塑形的上一步势能
        self._last_phi: float = 0.0
        self._pending_reset_reward: float = 0.0

    # 势函数 Phi(s)：基于最后叫点和己方手牌的成功期望，范围约 [-1, 1]
    def _phi(self, raw_obs: Dict) -> float:
        try:
            last_guess = raw_obs.get("last_guess")
            if last_guess is None:
                return 0.0
            # 兼容 dict 或 Guess
            mode = last_guess.get("mode") if isinstance(last_guess, dict) else getattr(last_guess, "mode", '飞')
            count = last_guess.get("count") if isinstance(last_guess, dict) else getattr(last_guess, "count", 0)
            face = last_guess.get("face") if isinstance(last_guess, dict) else getattr(last_guess, "face", 1)
            face = int(face)

            my_counts = raw_obs["my_dice_counts"]  # 长度6: 1..6
            c1, c2, c3, c4, c5, c6 = my_counts
            own_face = [0, c1, c2, c3, c4, c5, c6][face]
            own_ones = c1

            if mode == '飞':
                # 成功概率：P(face 或 1)；若 face==1 亦视为 2/6
                p = (2.0 / 6.0)
                own_succ = float(own_face + own_ones) if face != 1 else float(own_ones + own_face)
            else:  # 斋
                p = (1.0 / 6.0)
                own_succ = float(own_face)

            unknown = max(0, (self.num_players - 1) * self.dice_per_player)
            exp_total = own_succ + unknown * p
            gap = float(exp_total - float(count))
            return float(np.clip(gap / max(1, self._total_dice), -1.0, 1.0))
        except Exception:
            return 0.0

    # Helpers
    def _build_obs_for_agent(self, agent_id: str) -> Dict[str, np.ndarray]:
        assert self._env is not None
        obs_dict = self._env.observe(agent_id)
        state_vec = self._encoder.encode_observation(obs_dict)
        mask = self._env.get_action_mask(obs_dict)
        # 训练期软约束：在回合早期（叫点数少于阈值）隐藏 Challenge 动作，鼓励先进行加注博弈
        if (
            agent_id == self._rl_agent_id
            and self.early_challenge_min_raises > 0
            and len(self._env.round_history) < self.early_challenge_min_raises
        ):
            # 仅在存在其他合法 Guess 动作时，才隐藏 Challenge，避免掩码全False
            if mask.shape[0] > 1 and np.any(mask[1:]):
                mask[0] = False
        return {"obs": state_vec.astype(np.float32), "action_mask": mask.astype(np.int8)}

    def _step_opponents_until_rl_turn_or_done(self) -> Tuple[float, bool, bool]:
        total_reward = 0.0
        terminated = False
        truncated = False
        assert self._env is not None

        while self._env.agents and self._env.agent_selection != self._rl_agent_id:
            ag_id = self._env.agent_selection
            opp = self._seat_opponents.get(ag_id)
            if opp is None:
                opp = self.pool.sample()
                self._seat_opponents[ag_id] = opp

            if isinstance(opp, PolicyOpponent):
                enc_obs = self._build_obs_for_agent(ag_id)
                action_id = opp.predict_action_id(enc_obs)
                action = self._env.action_to_object(action_id)
            else:
                raw_obs = self._env.observe(ag_id)
                action = opp.get_action(raw_obs)

            self._env.step(action)
            total_reward += float(self._env.rewards.get(self._rl_agent_id, 0))
            terminated = all(self._env.terminations.get(a, False) for a in self._env.agents)
            truncated = all(self._env.truncations.get(a, False) for a in self._env.agents)
            if terminated or truncated:
                break
        return total_reward, terminated, truncated

    # Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._env = LiarDiceEnv(
            num_players=self.num_players,
            dice_per_player=self.dice_per_player,
            render_mode=self.render_mode,
            use_specialized_action_space=True,
        )
        self._env.reset(seed=seed)

        def reseat_opponents() -> List[str]:
            self._seat_opponents = {}
            descriptions: List[str] = []
            for i in range(self.num_players):
                aid = f"player_{i}"
                if aid == self._rl_agent_id:
                    continue
                opponent = self.pool.sample()
                self._seat_opponents[aid] = opponent
                if hasattr(opponent, 'get_description'):
                    descriptions.append(opponent.get_description())
                else:
                    descriptions.append("未知对手")
            return descriptions

        opponent_descriptions = reseat_opponents()

        if opponent_descriptions and getattr(self, '_show_opponent_info', False):
            print(f"对手配置: {', '.join(opponent_descriptions)}")

        self._pending_reset_reward = 0.0
        first_seed = seed
        attempts = 0
        max_attempts = 64

        while True:
            reward, terminated, truncated = self._step_opponents_until_rl_turn_or_done()
            self._pending_reset_reward += reward

            if not (terminated or truncated):
                if self._env.agent_selection == self._rl_agent_id:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Opponent stepping did not yield RL agent turn;"
                        f" attempts={attempts}, agent_selection={self._env.agent_selection}"
                    )
                continue

            attempts += 1
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Unable to reach RL agent turn after repeated self-play resets;"
                    f" attempts={attempts}, max_attempts={max_attempts},"
                    f" agent_selection={self._env.agent_selection},"
                    f" term_all={all(self._env.terminations.values())},"
                    f" trunc_all={all(self._env.truncations.values())}"
                )

            self._env.reset(seed=first_seed)
            first_seed = None
            # 清空上一局奖励，避免跨局累计
            self._pending_reset_reward = 0.0
            opponent_descriptions = reseat_opponents()
            if opponent_descriptions and getattr(self, '_show_opponent_info', False):
                print(f"对手配置: {', '.join(opponent_descriptions)}")

        # 初始化势能（用于塑形）
        raw_obs = self._env.observe(self._rl_agent_id)
        self._last_phi = self._phi(raw_obs) if self.dense_shaping else 0.0
        obs = self._build_obs_for_agent(self._rl_agent_id)
        info = {}
        return obs, info

    def step(self, action: int):
        assert self._env is not None
        assert self._env.agent_selection == self._rl_agent_id
        action_obj = self._env.action_to_object(action)
        # 记录 RL 动作前的历史长度，用于判定“过早挑战”
        history_len_before = len(self._env.round_history)
        self._env.step(action_obj)

        reward_rl = self._pending_reset_reward + float(self._env.rewards.get(self._rl_agent_id, 0))
        self._pending_reset_reward = 0.0
        terminated = all(self._env.terminations.get(a, False) for a in self._env.agents)
        truncated = all(self._env.truncations.get(a, False) for a in self._env.agents)

        if not (terminated or truncated):
            r2, term2, trunc2 = self._step_opponents_until_rl_turn_or_done()
            reward_rl += r2
            terminated = term2 or terminated
            truncated = trunc2 or truncated

        # 训练期行为塑形：
        # 1) 过早挑战的软惩罚：若历史叫点不足阈值且本次动作为 Challenge，则扣少量 shaping 分。
        # 2) 合法加注的小额奖励：若 RL 本次选择的是 Guess，则给予小额 shaping 奖励。
        try:
            from env import Challenge as _Challenge, Guess as _Guess  # 仅作类型比较
        except Exception:
            _Challenge, _Guess = None, None
        if _Challenge is not None and isinstance(action_obj, _Challenge):
            if history_len_before < self.early_challenge_min_raises and self.early_challenge_penalty > 0.0:
                reward_rl -= self.early_challenge_penalty
        elif _Guess is not None and isinstance(action_obj, _Guess):
            if self.guess_step_bonus > 0.0:
                reward_rl += self.guess_step_bonus

        # 潜在塑形：β * (γ * Phi(s') - Phi(s))
        if self.dense_shaping:
            if terminated or truncated:
                phi_sp = 0.0
            else:
                raw_obs_sp = self._env.observe(self._rl_agent_id)
                phi_sp = self._phi(raw_obs_sp)
            reward_rl += self.shaping_beta * (self.shaping_gamma * phi_sp - self._last_phi)
            self._last_phi = phi_sp

        if terminated or truncated:
            obs = self.observation_space.sample()
        else:
            obs = self._build_obs_for_agent(self._rl_agent_id)
        info = {}
        return obs, reward_rl, terminated, truncated, info

    def render(self):
        if self._env is not None:
            self._env.render()

    def close(self):
        if self._env is not None:
            self._env = None
