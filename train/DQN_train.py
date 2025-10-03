import math
import os
import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from env import LiarDiceEnv, Challenge, Action
from agents.DQN_agent import DQNAgent
from agents.heuristic_agent import HeuristicRuleAgent
from utils import get_legal_actions


def _build_action_masks(agent: DQNAgent, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return guess mask [B,2,C,6] and challenge legality [B] from state vectors."""

    batch_size = state_batch.shape[0]
    guess_masks = []
    challenge_legals = []

    zero_state = torch.zeros(agent.state_dim, device=state_batch.device)

    for idx in range(batch_size):
        state_vec = state_batch[idx]
        if torch.allclose(state_vec, zero_state):
            guess_masks.append(torch.zeros((2, agent.total_dice, 6), dtype=torch.bool, device=state_batch.device))
            challenge_legals.append(False)
            continue

        mask = agent.build_guess_legal_mask(state_vec)
        guess_masks.append(mask)
        challenge_legals.append(agent.decode_last_guess_from_state(state_vec) is not None)

    guess_mask_tensor = torch.stack(guess_masks, dim=0)
    challenge_tensor = torch.tensor(challenge_legals, dtype=torch.bool, device=state_batch.device)

    return guess_mask_tensor, challenge_tensor


def _soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
    """Polyak averaging of network parameters."""

    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def _binom_tail(n: int, p: float, r: int) -> float:
    """Compute P[X >= r] for X~Bin(n,p)."""

    if r <= 0:
        return 1.0
    if r > n:
        return 0.0

    q = 1.0 - p
    prob = 0.0
    for k in range(r, n + 1):
        prob += math.comb(n, k) * (p ** k) * (q ** (n - k))
    return float(min(max(prob, 0.0), 1.0))


def _potential_from_state(agent: DQNAgent, state_tensor: torch.Tensor, beta: float) -> float:
    """Potential based on estimated truth probability of the current last guess."""

    if beta <= 0.0:
        return 0.0

    state_vec = state_tensor.squeeze(0)
    last_guess = agent.decode_last_guess_from_state(state_vec)
    if last_guess is None:
        return 0.0

    my_counts = state_vec[:6].cpu().numpy()
    total_known = float(my_counts.sum())
    total_dice = agent.total_dice
    unknown = max(0, int(round(total_dice - total_known)))

    ones = my_counts[0]
    face_idx = last_guess.face - 1
    face_count = my_counts[face_idx]

    if last_guess.mode == '飞':
        own = face_count + ones if last_guess.face != 1 else ones
        success_prob = 2.0 / 6.0
    else:
        own = face_count
        success_prob = 1.0 / 6.0

    need = max(0, int(math.ceil(last_guess.count - own)))
    prob_true = _binom_tail(unknown, success_prob, need)

    return beta * (prob_true - 0.5)

class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, discount):
        """Save a transition"""
        self.buffer.append((state, action, reward, next_state, done, discount))

    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(agent: DQNAgent, observation: Dict, epsilon: float,
                 legal_actions: List[Action], *, round_step: int,
                 challenge_suppress_steps: int) -> Action:
    """Select action using epsilon-greedy policy.

    During exploration, for the first K steps of a round, suppress Challenge
    to encourage longer interactions. Exploitation path is unchanged.
    """
    is_exploring = (random.random() < epsilon)
    if is_exploring:
        candidates = legal_actions
        if challenge_suppress_steps > 0 and round_step < challenge_suppress_steps and observation.get('last_guess') is not None:
            filtered = [a for a in legal_actions if not isinstance(a, Challenge)]
            if len(filtered) > 0:
                candidates = filtered
        return random.choice(candidates)
    # Exploitation
    return agent.get_action(observation)


def make_frozen_opponent(agent: DQNAgent, args, *, agent_id: Optional[str] = None) -> DQNAgent:
    """Create a frozen copy of the learning agent for self-play opponents."""

    frozen = DQNAgent(
        agent_id=agent_id or agent.agent_id,
        num_players=agent.num_players,
        args=args,
    )
    frozen.q_network.load_state_dict(agent.q_network.state_dict())
    frozen.target_network.load_state_dict(agent.target_network.state_dict())
    frozen.q_network.eval()
    frozen.target_network.eval()
    return frozen


def sample_assignment(env: LiarDiceEnv, agent: DQNAgent, heuristic_ratio: float) -> Dict[str, str]:
    """Sample opponent type for each non-learning seat."""

    assignment: Dict[str, str] = {}
    for pid in env.possible_agents:
        if pid == agent.agent_id:
            continue
        assignment[pid] = "heuristic" if random.random() < heuristic_ratio else "frozen"
    if assignment and all(policy == "frozen" for policy in assignment.values()):
        random_pid = random.choice(list(assignment.keys()))
        assignment[random_pid] = "heuristic"
    return assignment


def quick_evaluate(agent: DQNAgent, num_players: int, episodes: int, opp_conf: int) -> float:
    """Run lightweight evaluation against heuristic opponents, returning win rate."""

    if episodes <= 0:
        return 0.0

    eval_env = LiarDiceEnv(num_players=num_players)
    opponents = {
        pid: HeuristicRuleAgent(pid, num_players, confidence_threshold=opp_conf)
        for pid in eval_env.possible_agents
        if pid != agent.agent_id
    }

    wins = 0
    for _ in range(episodes):
        eval_env.reset()
        done = False

        while not done:
            current = eval_env.agent_selection
            observation = eval_env.observe(current)

            if eval_env.terminations[current] or eval_env.truncations[current]:
                eval_env.step(None)
                continue

            if current == agent.agent_id:
                action = agent.get_action(observation)
            else:
                action = opponents[current].get_action(observation)

            eval_env.step(action)
            done = all(eval_env.terminations.values()) or all(eval_env.truncations.values())

        if eval_env.penalties[agent.agent_id] == min(eval_env.penalties.values()):
            wins += 1

    return wins / float(episodes)

def train_dqn(agent: DQNAgent, env: LiarDiceEnv, args) -> Tuple[DQNAgent, Dict]:
    """Complete DQN training pipeline"""

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    use_wandb = bool(getattr(args, "use_wandb", False))
    wandb_run = None
    if use_wandb:
        wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
        if wandb is None or wandb_disabled:
            print("⚠️  wandb unavailable or disabled; proceeding without online logging.")
            use_wandb = False
        else:
            wandb_run = wandb.init(
                entity="HCCS",
                project="liar_dice",
                config=vars(args)
            )

    # Initialize epsilon and step counter
    epsilon = args.epsilon_start
    global_step = 0

    print("🚀 Starting DQN Training...")
    print(f"📊 Training settings:")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gamma: {args.gamma}")
    print(f"   Epsilon: {args.epsilon_start} → {args.epsilon_end}")
    print(f"   Target update frequency (steps): {args.target_update_freq}")
    print(f"   Target soft tau: {getattr(args, 'target_soft_tau', 0.0)}")
    print(f"   n-step return: {getattr(args, 'n_step', 1)}")
    print(f"   Potential shaping beta: {getattr(args, 'shaping_beta', 0.0)}")
    print(f"   Rounds per episode: {getattr(args, 'rounds_per_episode', 1)}")
    print(f"   Updates per step (batched at round end): {getattr(args, 'updates_per_step', 1)}")
    print(f"   Warmup steps: {getattr(args, 'warmup_steps', 0)}")
    print(f"   Challenge suppress steps: {getattr(args, 'challenge_suppress_steps', 0)}")
    print(f"   Opponent confidence: {getattr(args, 'opponent_conf_min', 0)} → {args.opponent_confidence}")
    print(f"   Curriculum warmup: {getattr(args, 'curriculum_warmup', 0)} episodes")

    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)

    frozen_pool: List[DQNAgent] = []

    def refresh_frozen_pool() -> None:
        frozen_copy = make_frozen_opponent(agent, args, agent_id="frozen")
        frozen_pool.append(frozen_copy)
        pool_limit = max(1, getattr(args, "frozen_pool_size", 1))
        if len(frozen_pool) > pool_limit:
            frozen_pool.pop(0)

    refresh_frozen_pool()

    min_epsilon = max(getattr(args, "min_epsilon", 0.0), args.epsilon_end)

    for episode in tqdm(range(args.num_episodes), desc="Training Episodes"):
        episode_reward = 0.0
        episode_length = 0
        episode_challenge_cnt = 0
        episode_early_challenge_cnt = 0

        curriculum_progress = min(
            1.0,
            episode / max(1, getattr(args, 'curriculum_warmup', 1))
        )
        base_conf = getattr(args, 'opponent_conf_min', args.opponent_confidence)
        current_conf = int(round(
            (1.0 - curriculum_progress) * base_conf
            + curriculum_progress * args.opponent_confidence
        ))
        current_conf = max(0, current_conf)

        base_interval = max(1, getattr(args, 'frozen_opponent_update_interval', 1))
        growth_factor = max(1.0, getattr(args, 'frozen_interval_growth', 1.0))
        effective_interval = max(
            1,
            int(round(base_interval * (1.0 + curriculum_progress * (growth_factor - 1.0))))
        )

        if effective_interval > 0 and (episode % effective_interval == 0):
            refresh_frozen_pool()

        opponents = {
            pid: HeuristicRuleAgent(pid, env.num_players, confidence_threshold=current_conf)
            for pid in env.possible_agents
            if pid != agent.agent_id
        }

        episode_assignment = None
        if getattr(args, "mix_mode", "round") == "episode":
            episode_assignment = sample_assignment(env, agent, getattr(args, "heuristic_ratio", 0.5))

        rounds = getattr(args, 'rounds_per_episode', 1)
        for _ in range(rounds):
            env.reset()
            done = False
            round_step = 0
            step_data: List[Dict] = []

            if getattr(args, "mix_mode", "round") == "round":
                assignment = sample_assignment(env, agent, getattr(args, "heuristic_ratio", 0.5))
            else:
                assignment = episode_assignment or sample_assignment(env, agent, getattr(args, "heuristic_ratio", 0.5))

            while not done and round_step < args.max_steps_per_episode:
                current_agent_id = env.agent_selection
                if current_agent_id is None:
                    break

                observation = env.observe(current_agent_id)

                if env.terminations[current_agent_id] or env.truncations[current_agent_id]:
                    env.step(None)
                    continue

                if current_agent_id == agent.agent_id:
                    legal_actions = get_legal_actions(observation, args.num_players)
                    if not legal_actions:
                        raise RuntimeError(
                            f"No legal actions available for {current_agent_id}. last_guess={observation.get('last_guess')}"
                        )

                    suppress_steps = getattr(args, 'challenge_suppress_steps', 0)
                    if getattr(args, 'shaping_beta', 0.0) > 0:
                        suppress_steps = 0

                    action = select_action(
                        agent,
                        observation,
                        epsilon,
                        legal_actions,
                        round_step=round_step,
                        challenge_suppress_steps=suppress_steps
                    )

                    state_tensor = agent.get_state_vector(observation).unsqueeze(0)

                    if isinstance(action, Challenge):
                        main_action_idx = 1
                        mode_idx = 0
                        count_idx = 0
                        face_idx = 0
                        episode_challenge_cnt += 1
                        if round_step < getattr(args, 'challenge_suppress_steps', 0):
                            episode_early_challenge_cnt += 1
                    else:
                        main_action_idx = 0
                        mode_idx = 0 if action.mode == '飞' else 1
                        count_idx = action.count - 1
                        face_idx = action.face - 1

                    action_indices = torch.tensor([main_action_idx, mode_idx, count_idx, face_idx], dtype=torch.long)

                    env.step(action)

                    reward = env.rewards[current_agent_id]
                    termination = env.terminations[current_agent_id]
                    truncation = env.truncations[current_agent_id]

                    step_data.append({
                        'state': state_tensor,
                        'action': action_indices,
                        'reward': reward,
                        'done': termination or truncation,
                    })

                    episode_reward += reward
                    episode_length += 1
                    round_step += 1
                    global_step += 1

                    epsilon = max(min_epsilon, epsilon * args.epsilon_decay)

                    tau = getattr(args, 'target_soft_tau', 0.0)
                    if tau and tau > 0:
                        _soft_update(agent.target_network, agent.q_network, tau)
                    elif args.target_update_freq > 0 and (global_step % args.target_update_freq == 0):
                        agent.target_network.load_state_dict(agent.q_network.state_dict())

                    if termination or truncation:
                        done = True
                else:
                    policy = assignment.get(current_agent_id, "heuristic") if assignment else "heuristic"
                    if policy == "heuristic" or not frozen_pool:
                        action = opponents[current_agent_id].get_action(observation)
                    else:
                        frozen_opponent = random.choice(frozen_pool)
                        action = frozen_opponent.get_action(observation)

                    env.step(action)

                    if step_data:
                        final_reward = env.rewards[agent.agent_id]
                        last_reward = step_data[-1]['reward']
                        if final_reward != last_reward:
                            episode_reward += (final_reward - last_reward)
                            step_data[-1]['reward'] = final_reward

                    termination = env.terminations[current_agent_id]
                    truncation = env.truncations[current_agent_id]

                    if termination or truncation:
                        done = True

                if done and step_data:
                    final_reward = env.rewards[agent.agent_id]
                    last_reward = step_data[-1]['reward']
                    if final_reward != last_reward:
                        episode_reward += (final_reward - last_reward)
                        step_data[-1]['reward'] = final_reward
                    step_data[-1]['done'] = True

            if step_data:
                shaping_beta = getattr(args, 'shaping_beta', 0.0)
                if shaping_beta > 0:
                    potentials = [
                        _potential_from_state(agent, data['state'], shaping_beta)
                        for data in step_data
                    ]
                    potentials.append(0.0)
                    for idx, data in enumerate(step_data):
                        phi_s = potentials[idx]
                        phi_next = potentials[idx + 1] if idx + 1 < len(potentials) else 0.0
                        shaping_reward = args.gamma * phi_next - phi_s
                        data['reward'] += shaping_reward

                n_step = max(1, getattr(args, 'n_step', 1))
                gamma = args.gamma
                zero_state = torch.zeros_like(step_data[0]['state'])

                for i in range(len(step_data)):
                    reward_sum = 0.0
                    steps_taken = 0
                    done_flag = False

                    for j in range(n_step):
                        idx = i + j
                        if idx >= len(step_data):
                            break

                        reward_sum += (gamma ** j) * step_data[idx]['reward']
                        steps_taken += 1

                        if step_data[idx].get('done', False):
                            done_flag = True
                            break

                    if not done_flag and (i + steps_taken) < len(step_data):
                        next_state = step_data[i + steps_taken]['state']
                        discount = gamma ** steps_taken
                    else:
                        next_state = zero_state
                        discount = 0.0

                    replay_buffer.push(
                        step_data[i]['state'],
                        step_data[i]['action'],
                        reward_sum,
                        next_state,
                        done_flag,
                        discount
                    )

                updates = getattr(args, 'updates_per_step', 1) * len(step_data)
                for _ in range(updates):
                    if global_step >= getattr(args, 'warmup_steps', 0) and len(replay_buffer) >= args.batch_size:
                        loss = train_agent_step(agent, replay_buffer, args)
                        if use_wandb:
                            wandb.log({"loss_values": loss}, step=global_step)

        eval_interval = getattr(args, "eval_interval", 0)
        eval_episodes = getattr(args, "eval_episodes", 0)
        if eval_interval > 0 and eval_episodes > 0 and (episode + 1) % eval_interval == 0:
            win_rate = quick_evaluate(agent, args.num_players, eval_episodes, current_conf)
            print(
                f"[Eval] episode={episode + 1} win_rate_vs_heuristic(conf={current_conf}): {win_rate * 100:.1f}%"
            )

        if (
            getattr(args, "epsilon_reset_interval", 0) > 0
            and (episode + 1) % args.epsilon_reset_interval == 0
        ):
            reset_value = max(min_epsilon, getattr(args, "epsilon_reset_value", args.epsilon_start))
            epsilon = max(min_epsilon, reset_value)

        # Episode-level logging
        if use_wandb:
            wandb.log({
                "episode_rewards": episode_reward,
                "episode_lengths": episode_length,
                "epsilon_values": epsilon,
                "challenge_rate": (episode_challenge_cnt / max(1, episode_length)),
                "early_challenge_rate": (episode_early_challenge_cnt / max(1, episode_challenge_cnt)),
                "replay_buffer_size": len(replay_buffer)
            }, step=episode)

        # Save model per episode
        if episode % args.save_model_freq == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': epsilon,
                'num_players': args.num_players,
                'learning_rate': args.learning_rate,
            }, args.model_save_path.replace('.pth', f'_ep{episode}.pth'))

    # Save final model
    torch.save({
        'episode': args.num_episodes,
        'model_state_dict': agent.q_network.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': epsilon,
        'num_players': args.num_players,
        'learning_rate': args.learning_rate,
    }, args.model_save_path)

    print(f"✅ Training completed! Model saved to {args.model_save_path}")

    if wandb_run is not None:
        wandb_run.finish()

    return agent

def train_agent_step(agent: DQNAgent, replay_buffer: ReplayBuffer, args) -> float:
    """Train agent for one step using experience replay"""

    # Sample batch from replay buffer
    batch = replay_buffer.sample(args.batch_size)

    # Convert batch to tensors
    states = torch.cat([data[0] for data in batch])
    actions = torch.stack([data[1] for data in batch])  # [batch_size, 4]
    rewards = torch.tensor([data[2] for data in batch], dtype=torch.float32)
    next_states = torch.cat([data[3] for data in batch])
    dones = torch.tensor([data[4] for data in batch], dtype=torch.float32)
    discounts = torch.tensor([data[5] for data in batch], dtype=torch.float32)

    # Extract action indices
    main_actions = actions[:, 0]  # Main action indices
    mode_actions = actions[:, 1]   # Mode indices
    count_actions = actions[:, 2]  # Count indices
    face_actions = actions[:, 3]   # Face indices

    # Current Q values
    current_q_main, current_q_mode, current_q_count, current_q_face = agent.q_network(states)

    # Compose Q for executed actions
    challenge_q = current_q_main[:, 1]
    guess_q = (
        current_q_main[:, 0]
        + current_q_mode.gather(1, mode_actions.unsqueeze(1)).squeeze(1)
        + current_q_count.gather(1, count_actions.unsqueeze(1)).squeeze(1)
        + current_q_face.gather(1, face_actions.unsqueeze(1)).squeeze(1)
    )

    is_challenge = main_actions == 1
    chosen_q = torch.where(is_challenge, challenge_q, guess_q)

    with torch.no_grad():
        next_q_main_t, next_q_mode_t, next_q_count_t, next_q_face_t = agent.target_network(next_states)
        next_q_main_o, next_q_mode_o, next_q_count_o, next_q_face_o = agent.q_network(next_states)

        guess_mask, challenge_legal = _build_action_masks(agent, next_states)
        flat_mask = guess_mask.view(guess_mask.size(0), -1)
        has_legal_guess = flat_mask.any(dim=1)

        # Online selection
        guess_q_online = (
            next_q_main_o[:, 0].view(-1, 1, 1, 1)
            + next_q_mode_o.view(-1, 2, 1, 1)
            + next_q_count_o.view(-1, 1, agent.total_dice, 1)
            + next_q_face_o.view(-1, 1, 1, 6)
        ).masked_fill(~guess_mask, float('-inf'))

        guess_q_online_flat = guess_q_online.view(guess_q_online.size(0), -1)
        best_guess_vals_online, best_guess_idx = guess_q_online_flat.max(dim=1)
        best_guess_vals_online = torch.where(
            has_legal_guess,
            best_guess_vals_online,
            torch.full_like(best_guess_vals_online, float('-inf'))
        )
        best_guess_idx = torch.where(
            has_legal_guess,
            best_guess_idx,
            torch.zeros_like(best_guess_idx)
        )

        challenge_vals_online = next_q_main_o[:, 1]
        challenge_vals_online = torch.where(
            challenge_legal,
            challenge_vals_online,
            torch.full_like(challenge_vals_online, float('-inf'))
        )

        pick_challenge = challenge_vals_online >= best_guess_vals_online

        # Target evaluation
        guess_q_target = (
            next_q_main_t[:, 0].view(-1, 1, 1, 1)
            + next_q_mode_t.view(-1, 2, 1, 1)
            + next_q_count_t.view(-1, 1, agent.total_dice, 1)
            + next_q_face_t.view(-1, 1, 1, 6)
        ).masked_fill(~guess_mask, float('-inf'))

        best_guess_target = guess_q_target.view(guess_q_target.size(0), -1).gather(
            1, best_guess_idx.unsqueeze(1)
        ).squeeze(1)
        best_guess_target = torch.where(
            has_legal_guess,
            best_guess_target,
            torch.full_like(best_guess_target, float('-inf'))
        )

        best_next_q = torch.where(
            pick_challenge,
            next_q_main_t[:, 1],
            best_guess_target
        )

        best_next_q = torch.where(
            torch.isfinite(best_next_q),
            best_next_q,
            torch.zeros_like(best_next_q)
        )

    targets = rewards + (1 - dones) * discounts * best_next_q

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(chosen_q, targets)

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=1.0)
    agent.optimizer.step()

    return loss.item()
