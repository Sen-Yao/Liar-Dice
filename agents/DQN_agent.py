import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict
import sys

# Ensure project root on sys.path when running this module directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import Guess, Challenge, Action, LiarDiceEnv

class ParametricQNetwork(nn.Module):
    # 根据状态向量的输入, 判断各个动作的 Q 值
    # 原始输入通过一个共享的主干网络, 得到一个高维的特征向量.
    # 特征向量分别经过四个头, 来得到对应的动作
    # 最后各个动作的 Q ☞将会返回
    def __init__(self, state_dim, num_players, feature_dim):
        super(ParametricQNetwork, self).__init__()
        self.num_players = num_players
        total_dice = 5 * self.num_players

        # 共享主干网络 (Shared Trunk)
        self.shared_trunk = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )

        # 这个头用来区分两个宏观动作：“猜测” 和 “检验”
        self.main_action_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出维度为2，分别代表 Q(s, Bid) 和 Q(s, Challenge)
        )


        # Mode Head
        self.mode_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 0: 飞, 1: 斋
        )

        # Count Head
        self.count_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, total_dice) # 输出维度为总骰子数，索引对应数量-1
        )

        # Face Head
        self.face_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6) # 0-5 对应点数 1-6
        )

    def forward(self, state):
        # 通过主干网络提取特征
        features = self.shared_trunk(state)

        # 分别通过各个头计算 Q 值
        main_action_q_values = self.main_action_head(features)
        mode_q_values = self.mode_head(features)
        count_q_values = self.count_head(features)
        face_q_values = self.face_head(features)

        return main_action_q_values, mode_q_values, count_q_values, face_q_values

class DQNAgent:
    """
    一个基于 DQN 的 Agent
    """
    def __init__(self, agent_id: str, num_players: int, args, feature_dim=128,):
        self.agent_id = agent_id
        self.num_players = num_players
        self.state_dim = 19  # 从16增加到19（+3维历史统计特征）
        self.total_dice = 5 * self.num_players

        self.q_network = ParametricQNetwork(self.state_dim, num_players, feature_dim=feature_dim)
        self.target_network = ParametricQNetwork(self.state_dim, num_players, feature_dim=feature_dim)
        
        # 创建之初，必须将主网络的权重完全复制给 Target 网络，确保它们从同一起点开始
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 设置 Target 网络为评估模式，因为我们不用对它进行训练（反向传播）
        self.target_network.eval()

        # 定义优化器，只优化主网络的参数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def get_action(self, observation: Dict) -> Action:
        state_vec = self.get_state_vector(observation=observation)
        with torch.no_grad():
            main_q, mode_q, count_q, face_q = self.q_network(state_vec)

        device = state_vec.device
        counts = torch.arange(1, self.total_dice + 1, device=device)
        faces = torch.arange(1, 7, device=device)

        # 计算所有组合的 Q 值网格：shape = [2(modes), total_dice(counts), 6(faces)]
        q_grid = (main_q[0]
                  + mode_q.view(2, 1, 1)
                  + count_q.view(1, self.total_dice, 1)
                  + face_q.view(1, 1, 6))

        last_guess = observation['last_guess']

        legal_mask = torch.zeros_like(q_grid, dtype=torch.bool)
        counts_grid = counts.view(1, self.total_dice, 1)
        faces_grid = faces.view(1, 1, 6)

        if last_guess is None:
            # 首轮：count > 玩家数，飞模式禁止 face=1
            count_mask = (counts_grid > self.num_players)
            flight_mask = count_mask & (faces_grid != 1)

            legal_mask[0] = flight_mask.squeeze(0)
            legal_mask[1] = count_mask.expand_as(flight_mask).squeeze(0)
            q_challenge = torch.full((), -1e9, device=device)
        else:
            old_count = last_guess.count
            old_face = last_guess.face
            old_mode = 0 if last_guess.mode == '飞' else 1

            zhai_rank_lookup = torch.tensor([0, 5, 0, 1, 2, 3, 4], device=device)
            faces_rank = zhai_rank_lookup[faces.long()].view(1, 1, 6)
            old_rank = zhai_rank_lookup[old_face]

            # 同模式比较
            if old_mode == 0:  # 上一个是飞
                fly_mask = ((counts_grid > old_count) |
                            ((counts_grid == old_count) & (faces_grid > old_face)))
                fly_mask = fly_mask & (faces_grid != 1)
                legal_mask[0] = fly_mask.squeeze(0)

                ceil_half = (old_count + 1) // 2
                zhai_mask = (counts_grid >= ceil_half)
                legal_mask[1] = zhai_mask.expand_as(fly_mask).squeeze(0)
            else:  # 上一个是斋
                zhai_mask = ((counts_grid > old_count) |
                             ((counts_grid == old_count) & (faces_rank > old_rank)))
                legal_mask[1] = zhai_mask.squeeze(0)

                fly_mask = (counts_grid >= (old_count * 2)) & (faces_grid != 1)
                legal_mask[0] = fly_mask.squeeze(0)

            q_challenge = main_q[1]

        if not legal_mask.any():
            # No legal guess remains; by rules the only option is to challenge.
            return Challenge()

        masked_scores = q_grid.masked_fill(~legal_mask, -1e9)
        best_flat_idx = masked_scores.view(-1).argmax()
        best_mode_idx = (best_flat_idx // (self.total_dice * 6)).item()
        remainder = best_flat_idx % (self.total_dice * 6)
        best_count_idx = (remainder // 6).item()
        best_face_idx = (remainder % 6).item()

        best_guess_q = masked_scores[best_mode_idx, best_count_idx, best_face_idx]

        if q_challenge.item() > best_guess_q.item():
            return Challenge()

        mode = '飞' if best_mode_idx == 0 else '斋'
        count = best_count_idx + 1
        face = best_face_idx + 1
        return Guess(mode=mode, count=count, face=face)

    def get_state_vector(self, observation: Dict):
        """将观测转换为状态向量（16→19维）

        原有16维特征：
        - my_dice_counts: 6维，手牌统计
        - num_players: 1维，玩家总数
        - last_guess_num: 1维，上次猜测个数（归一化）
        - last_guess_face: 6维，上次猜测点数（one-hot）
        - last_guess_mode: 2维，上次猜测模式（one-hot）

        新增3维历史统计特征：
        - current_round_progress: 当前轮次进度（history长度/总骰子数）
        - history_avg_count: 本轮历史平均猜测个数（归一化）
        - history_max_count: 本轮历史最大猜测个数（归一化）

        总计：6+1+1+6+2+1+1+1 = 19维

        删除的冗余特征：
        - challenge_urgency（与 last_guess_num 完全相同）
        - num_alive_players_ratio（硬编码为1.0，无信息）
        """
        # 原有16维特征
        my_dice_counts = torch.tensor(list(observation['my_dice_counts']), dtype=torch.float32)
        num_players = torch.tensor([self.num_players], dtype=torch.float32)

        last_guess = observation["last_guess"]
        last_guess_face = [0.0] * 6

        if last_guess is None:
            last_guess_mode = torch.tensor([0.0, 0.0], dtype=torch.float32)
            last_guess_num = torch.zeros(1, dtype=torch.float32)
        else:
            last_guess_face[last_guess.face - 1] = 1.0
            last_guess_num = torch.tensor(
                [last_guess.count / (self.num_players * 5.0)],
                dtype=torch.float32
            )

            if last_guess.mode == '飞':
                last_guess_mode = torch.tensor([1.0, 0.0], dtype=torch.float32)
            else:
                last_guess_mode = torch.tensor([0.0, 1.0], dtype=torch.float32)

        last_guess_face = torch.tensor(last_guess_face, dtype=torch.float32)

        # 新增3维历史统计特征
        history = observation.get('game_round_history', [])
        total_dice = self.num_players * 5.0

        if len(history) > 0:
            # 当前轮次进度
            current_round_progress = torch.tensor([len(history) / total_dice], dtype=torch.float32)

            # 历史猜测个数统计
            history_counts = [guess.count for _, guess in history]
            history_avg_count = torch.tensor([np.mean(history_counts) / total_dice], dtype=torch.float32)
            history_max_count = torch.tensor([max(history_counts) / total_dice], dtype=torch.float32)
        else:
            current_round_progress = torch.zeros(1, dtype=torch.float32)
            history_avg_count = torch.zeros(1, dtype=torch.float32)
            history_max_count = torch.zeros(1, dtype=torch.float32)

        # 拼接所有特征：6+1+1+6+2+1+1+1 = 19维
        state_vector = torch.cat([
            my_dice_counts,              # 6维：手牌统计
            num_players,                  # 1维：玩家总数
            last_guess_num,              # 1维：上次猜测个数
            last_guess_face,             # 6维：上次猜测点数
            last_guess_mode,             # 2维：上次猜测模式
            current_round_progress,      # 1维：轮次进度
            history_avg_count,           # 1维：历史平均猜测
            history_max_count            # 1维：历史最大猜测
        ], dim=0)

        return state_vector.to(dtype=torch.float32)

    def _create_masks(self, last_guess):
        """
        根据当前观测值，为所有动作头创建掩码。
        掩码中 1 代表合法, 0 代表非法。

        Args:
            last_guess: 上一个猜测

        Returns:
            Tuple[torch.Tensor, ...]: 四个掩码，分别对应
                                       main_action_head, mode_head, count_head, face_head。
        """
        total_dice = 5 * self.num_players

        # 初始化掩码
        # 主动作: [猜测, 检验]
        main_action_mask = torch.zeros(2, dtype=torch.float32)
        # 模式: [飞, 斋] 
        mode_mask = torch.zeros(2, dtype=torch.float32)
        # 数量: [1, 2, ..., total_dice]
        count_mask = torch.zeros(total_dice, dtype=torch.float32)
        # 点数: [1, 2, 3, 4, 5, 6]
        face_mask = torch.zeros(6, dtype=torch.float32)

        # 处理首回合特殊情况
        if last_guess is None:
            # a. 必须“猜测”，不能“检验”
            main_action_mask[0] = 1  # 猜测合法
            main_action_mask[1] = 0  # 检验非法

            # b. 任何模式和点数都合法
            mode_mask[:] = 1
            face_mask[:] = 1

            # c. 猜测的数量必须严格大于玩家数量（> num_players）
            # count_mask的索引 i 对应数量 i+1
            # 要求 count > num_players，即索引从 num_players 开始（对应 count=num_players+1）
            count_mask[self.num_players:] = 1

            return main_action_mask, mode_mask, count_mask, face_mask

        # 处理非首回合的通用情况        
        # “检验”现在是合法动作
        main_action_mask[1] = 1
        
        # 遍历所有可能的“猜测”动作，找出所有合法的参数
        # mode_idx 0 -> '飞', 1 -> '斋'
        # count_idx 0 -> 1个, 1 -> 2个, ...
        # face_idx 0 -> 1点, 1 -> 2点, ...
        modes = ['飞', '斋'] 
        
        has_any_legal_guess = False
        for mode_idx, mode in enumerate(modes):
            for count_idx in range(total_dice):
                count = count_idx + 1
                for face_idx in range(6):
                    face = face_idx + 1

                    # 规则检验 1: 飞模式不能喊 1
                    if mode == '飞' and face == 1:
                        continue  # 跳过这个非法的组合

                    current_guess = Guess(mode=mode, count=count, face=face)

                    # 规则检验 2: 新猜测必须严格大于上一个猜测
                    # 使用环境的规则作为单一真值来源
                    if LiarDiceEnv._is_strictly_greater(current_guess, last_guess):
                        # 如果这个猜测合法，那么它的所有组成部分都是合法的
                        mode_mask[mode_idx] = 1
                        count_mask[count_idx] = 1
                        face_mask[face_idx] = 1
                        has_any_legal_guess = True

        # 如果通过遍历发现没有任何合法的猜测（例如游戏结束或特殊规则），
        # 则“猜测”这个主动作变为非法。
        if has_any_legal_guess:
            main_action_mask[0] = 1
        else:
            main_action_mask[0] = 0 # 强制只能“检验”

        return main_action_mask, mode_mask, count_mask, face_mask
