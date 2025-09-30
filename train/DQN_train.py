import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import wandb

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
                 legal_actions: List[Action]) -> Action:
    """Select action using epsilon-greedy policy"""
    if random.random() < epsilon:
        # Exploration: choose random legal action
        return random.choice(legal_actions)
    else:
        # Exploitation: use agent's policy
        return agent.get_action(observation)

def train_dqn(agent: DQNAgent, env: LiarDiceEnv, args) -> Tuple[DQNAgent, Dict]:
    """Complete DQN training pipeline"""

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    run = wandb.init(
        entity="HCCS",
        # Set the wandb project where this run will be logged.
        project="liar_dice",
        # Track hyperparameters and run metadata.
        config=args,
    )

    # Initialize epsilon
    epsilon = args.epsilon_start

    print("🚀 Starting DQN Training...")
    print(f"📊 Training settings:")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gamma: {args.gamma}")
    print(f"   Epsilon: {args.epsilon_start} → {args.epsilon_end}")
    print(f"   Target update frequency: {args.target_update_freq}")

    # Create models directory if it doesn't exist
    import os
    os.makedirs("./models", exist_ok=True)

    for episode in tqdm(range(args.num_episodes), desc="Training Episodes"):
        # Reset environment
        env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        step_data = []  # Store transitions for this episode

        while not done and episode_length < args.max_steps_per_episode:
            # Get current agent (assuming we're training one agent at a time)
            current_agent_id = env.agent_selection

            # Get observation for current agent
            if current_agent_id is not None:
                observation = env.observe(current_agent_id)

                # Get legal actions
                legal_actions = get_legal_actions(observation, args.num_players)

                if legal_actions:  # If there are legal actions
                    # Select action
                    action = select_action(agent, observation, epsilon, legal_actions)

                    # Convert to tensor for storage
                    state_tensor = agent.get_state_vector(observation).unsqueeze(0)

                    # Convert action to indices for storage
                    if isinstance(action, Challenge):
                        main_action_idx = 1  # Challenge
                        mode_idx = 0  # dummy value
                        count_idx = 0  # dummy value
                        face_idx = 0  # dummy value
                    else:  # Guess
                        main_action_idx = 0  # Guess
                        mode_idx = 0 if action.mode == '飞' else 1
                        count_idx = action.count - 1  # 0-based index
                        face_idx = action.face - 1  # 0-based index

                    action_indices = torch.tensor([main_action_idx, mode_idx, count_idx, face_idx], dtype=torch.long)

                    # Step environment
                    env.step(action)

                    # Get reward and termination info
                    reward = env.rewards[current_agent_id]
                    termination = env.terminations[current_agent_id]
                    truncation = env.truncations[current_agent_id]

                    # Store transition
                    step_data.append({
                        'state': state_tensor,
                        'action': action_indices,
                        'reward': reward,
                        'next_state': None,  # Will be filled later
                        'done': termination or truncation
                    })

                    episode_reward += reward
                    episode_length += 1

                    # Check if episode is done
                    if termination or truncation:
                        done = True

                        # Fill next_state for all transitions
                        for i, data in enumerate(step_data):
                            if i < len(step_data) - 1:
                                data['next_state'] = step_data[i + 1]['state']
                            else:
                                data['next_state'] = agent.get_state_vector(observation).unsqueeze(0)

                            # Add to replay buffer
                            replay_buffer.push(
                                data['state'],
                                data['action'],
                                data['reward'],
                                data['next_state'],
                                data['done']
                            )
                else:
                    # No legal actions, skip
                    env.step(None)
            else:
                # Agent not available, step environment
                env.step(None)

        # Update statistics
        wandb.log({ "episode_rewards": episode_reward,
                    "episode_lengths": episode_length,
                    "epsilon_values": epsilon}, step=episode)

        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # Train agent if we have enough samples
        if len(replay_buffer) >= args.batch_size:
            loss = train_agent_step(agent, replay_buffer, args)
            wandb.log({ "loss_values": loss}, step=episode)

        # Update target network
        if episode % args.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
            # print(f"🔄 Target network updated at episode {episode}")

        # Save model
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