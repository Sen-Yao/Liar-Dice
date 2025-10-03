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

from env import LiarDiceEnv, Guess, Challenge, Action
from agents.DQN_agent import DQNAgent
from utils import is_strictly_greater, get_legal_actions

class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.buffer.append((state, action, reward, next_state, done))

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
    print(f"   Rounds per episode: {getattr(args, 'rounds_per_episode', 1)}")
    print(f"   Updates per step (batched at round end): {getattr(args, 'updates_per_step', 1)}")
    print(f"   Warmup steps: {getattr(args, 'warmup_steps', 0)}")
    print(f"   Challenge suppress steps: {getattr(args, 'challenge_suppress_steps', 0)}")

    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)

    for episode in tqdm(range(args.num_episodes), desc="Training Episodes"):
        episode_reward = 0.0
        episode_length = 0
        episode_challenge_cnt = 0
        episode_early_challenge_cnt = 0

        rounds = getattr(args, 'rounds_per_episode', 1)
        for _ in range(rounds):
            # New round
            env.reset()
            done = False
            round_step = 0
            step_data = []

            while not done and round_step < args.max_steps_per_episode:
                current_agent_id = env.agent_selection
                if current_agent_id is None:
                    break

                observation = env.observe(current_agent_id)

                if env.terminations[current_agent_id] or env.truncations[current_agent_id]:
                    env.step(None)
                    continue

                legal_actions = get_legal_actions(observation, args.num_players)
                if not legal_actions:
                    raise RuntimeError(
                        f"No legal actions available for {current_agent_id}. last_guess={observation.get('last_guess')}"
                    )

                action = select_action(
                    agent,
                    observation,
                    epsilon,
                    legal_actions,
                    round_step=round_step,
                    challenge_suppress_steps=getattr(args, 'challenge_suppress_steps', 0)
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
                    'next_state': None,
                    'done': termination or truncation
                })

                # Step-level bookkeeping
                episode_reward += reward
                episode_length += 1
                round_step += 1
                global_step += 1

                # Epsilon decay per step
                epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

                # Periodic target update by steps
                if args.target_update_freq > 0 and (global_step % args.target_update_freq == 0):
                    agent.target_network.load_state_dict(agent.q_network.state_dict())

                if termination or truncation:
                    done = True

            # End of round: backfill next_state and push transitions
            if len(step_data) > 0:
                last_obs_for_next = env.observe(env.agent_selection) if env.agent_selection is not None else observation
                for i, data in enumerate(step_data):
                    if i < len(step_data) - 1:
                        data['next_state'] = step_data[i + 1]['state']
                    else:
                        # Use the latest available observation as terminal next state proxy
                        data['next_state'] = agent.get_state_vector(last_obs_for_next).unsqueeze(0)

                    replay_buffer.push(
                        data['state'],
                        data['action'],
                        data['reward'],
                        data['next_state'],
                        data['done']
                    )

                # Perform updates proportional to steps collected in this round
                updates = getattr(args, 'updates_per_step', 1) * len(step_data)
                for _ in range(updates):
                    if global_step >= getattr(args, 'warmup_steps', 0) and len(replay_buffer) >= args.batch_size:
                        loss = train_agent_step(agent, replay_buffer, args)
                        if use_wandb:
                            wandb.log({"loss_values": loss}, step=global_step)

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
                'epsilon': epsilon
            }, args.model_save_path.replace('.pth', f'_ep{episode}.pth'))

    # Save final model
    torch.save({
        'episode': args.num_episodes,
        'model_state_dict': agent.q_network.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': epsilon
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

    # Extract action indices
    main_actions = actions[:, 0]  # Main action indices
    mode_actions = actions[:, 1]   # Mode indices
    count_actions = actions[:, 2]  # Count indices
    face_actions = actions[:, 3]   # Face indices

    # Current Q values
    current_q_main, current_q_mode, current_q_count, current_q_face = agent.q_network(states)

    # Next Q values from target network
    with torch.no_grad():
        next_q_main, next_q_mode, next_q_count, next_q_face = agent.target_network(next_states)

    # Compute target Q values for main action (used for both actions)
    target_q_values = rewards + (1 - dones) * args.gamma * next_q_main.max(dim=1)[0]

    # Compute loss for each head
    # For main action: use target_q_values
    loss_main = nn.MSELoss()(current_q_main.gather(1, main_actions.unsqueeze(1)),
                            target_q_values.unsqueeze(1))

    # For other heads: use their own target values
    # Only train other heads when main action is "guess" (index 0)
    guess_mask = (main_actions == 0)

    if guess_mask.sum() > 0:  # If there are any guess actions
        # For mode, count, face heads, target is their own max next Q values
        target_mode = rewards + (1 - dones) * args.gamma * next_q_mode.max(dim=1)[0]
        target_count = rewards + (1 - dones) * args.gamma * next_q_count.max(dim=1)[0]
        target_face = rewards + (1 - dones) * args.gamma * next_q_face.max(dim=1)[0]

        # Only compute loss for guess actions
        loss_mode = nn.MSELoss()(
            current_q_mode[guess_mask].gather(1, mode_actions[guess_mask].unsqueeze(1)),
            target_mode[guess_mask].unsqueeze(1)
        )
        loss_count = nn.MSELoss()(
            current_q_count[guess_mask].gather(1, count_actions[guess_mask].unsqueeze(1)),
            target_count[guess_mask].unsqueeze(1)
        )
        loss_face = nn.MSELoss()(
            current_q_face[guess_mask].gather(1, face_actions[guess_mask].unsqueeze(1)),
            target_face[guess_mask].unsqueeze(1)
        )
    else:
        loss_mode = torch.tensor(0.0)
        loss_count = torch.tensor(0.0)
        loss_face = torch.tensor(0.0)

    # Total loss
    total_loss = loss_main + 0.3 * loss_mode + 0.3 * loss_count + 0.3 * loss_face

    # Optimize
    agent.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=1.0)
    agent.optimizer.step()

    return total_loss.item()
