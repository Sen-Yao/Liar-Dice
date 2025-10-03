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
    parser.add_argument("--max_steps_per_episode", type=int, default=200, help="每个 episode 的最大游戏步（按轮计）")
    parser.add_argument("--rounds_per_episode", type=int, default=10, help="每个 episode 里包含的轮数(round)")
    parser.add_argument("--batch_size", type=int, default=32, help="从缓冲区抽取的 Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Bellman 方程的折扣系数")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="启动时的贪婪 Epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="结束时的贪婪 Epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="按步衰减系数（每步乘以该系数）")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="按步更新 target 网络的频率（steps）")
    parser.add_argument("--updates_per_step", type=int, default=1, help="每个环境步期望执行的优化次数（按回合末成批执行）")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="仅收集数据的预热步数，期间不训练")
    parser.add_argument("--challenge_suppress_steps", type=int, default=2, help="每轮最开始的K步探索时禁止Challenge")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")

    # Environment settings
    parser.add_argument("--num_players", type=int, default=2, help="玩家个数")

    # Replay buffer
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="缓冲区大小")
    # Model saving
    parser.add_argument("--save_model_freq", type=int, default=100, help="模型保存间隔 episode")
    parser.add_argument("--model_save_path", type=str, default="./models/dqn_model.pth", help="模型保存路径")
    parser.add_argument("--use_wandb", action="store_true", help="启用 wandb 日志记录（默认关闭）")

    args = parser.parse_args()

    trained_agent = main(args)
