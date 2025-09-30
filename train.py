import argparse
from train.DQN_train import train_dqn
from env import LiarDiceEnv

from agents.DQN_agent import DQNAgent

def main(args):
    """Main training function"""
    # Create environment
    env = LiarDiceEnv(num_players=args.num_players)

    # Create DQN agent
    agent = DQNAgent(agent_id="player_0", num_players=args.num_players, args=args)

    # Train the agent
    trained_agent = train_dqn(agent, env, args)

    print("🎉 Training completed successfully!")
    return trained_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000, help="训练的 episode 数")
    parser.add_argument("--max_steps_per_episode", type=int, default=200, help="每个 episode 的最大游戏步")
    parser.add_argument("--batch_size", type=int, default=32, help="从缓冲区抽取的 Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Bellman 方程的折扣系数")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="启动时的贪婪 Epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="结束时的贪婪 Epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="贪婪 Epsilon 的衰减系数")
    parser.add_argument("--target_update_freq", type=int, default=100, help="多少个 episodes 更新一次模型")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")

    # Environment settings
    parser.add_argument("--num_players", type=int, default=2, help="玩家个数")

    # Replay buffer
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="缓冲区大小")
    # Model saving
    parser.add_argument("--save_model_freq", type=int, default=100, help="模型保存间隔 episode")
    parser.add_argument("--model_save_path", type=str, default="./models/dqn_model.pth", help="模型保存路径")

    args = parser.parse_args()

    trained_agent = main(args)