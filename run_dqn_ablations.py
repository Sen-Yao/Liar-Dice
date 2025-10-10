import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    """执行命令并原样打印完整输出。"""

    print("$", " ".join(cmd))
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if cp.stdout:
        print(cp.stdout)
    return cp


ABLATION_ARGS = {
    "baseline": [],
    "ddqn_off": ["--ddqn", "False"],
    "reward_shaping_off": ["--reward_shaping", "False"],
    "n_step_1": ["--n_step", "1"],
    "history_3": ["--history_len", "3"],
    "history_5": ["--history_len", "5"],
    "opponent_heuristic": ["--opponent_type", "heuristic"],
    "opponent_selfplay": ["--opponent_type", "selfplay"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行单个 DQN 消融实验并评估（配置与 train.py 默认一致）")
    parser.add_argument("--ablation", choices=ABLATION_ARGS.keys(), default="opponent_selfplay", help="选择要运行的实验")
    parser.add_argument("--num-episodes", type=int, default=1000, help="训练 episode 数，与 train.py 默认一致")
    parser.add_argument("--num-players", type=int, default=2, help="玩家数量，与 train.py 默认一致")
    parser.add_argument("--eval-games", type=int, default=300, help="评估对局数")
    parser.add_argument("--exp-tag", type=str, default="", help="自定义实验标签（留空自动生成）")
    args = parser.parse_args()

    extra_args = ABLATION_ARGS[args.ablation][:]

    if args.exp_tag:
        exp_tag = args.exp_tag
    else:
        exp_tag = f"dqn_{args.ablation}_{int(time.time())}"
        extra_args += ["--exp_tag", exp_tag]

    if "--exp_tag" not in extra_args:
        extra_args += ["--exp_tag", exp_tag]

    print("=== DQN 单次实验 ===")
    print(f"实验: {args.ablation}")
    print(f"Exp tag: {exp_tag}")
    print(f"训练 episodes: {args.num_episodes}, 玩家数: {args.num_players}")

    train_cmd = [
        sys.executable,
        "train.py",
        "--num_episodes",
        str(args.num_episodes),
        "--num_players",
        str(args.num_players),
    ] + extra_args

    train_proc = run_cmd(train_cmd)
    if train_proc.returncode != 0:
        sys.exit(train_proc.returncode)

    model_dir = Path("models") / exp_tag
    if not model_dir.exists():
        print(f"⚠️ 未找到模型目录: {model_dir}")
        sys.exit(1)

    eval_cmd = [
        sys.executable,
        "evaluator.py",
        "--model-dir",
        str(model_dir),
        "--num-games",
        str(args.eval_games),
        "--json",
    ]

    eval_proc = run_cmd(eval_cmd)
    if eval_proc.returncode != 0:
        sys.exit(eval_proc.returncode)

    try:
        payload = json.loads((eval_proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        payload = []

    out_dir = Path("runs") / "ablations_dqn" / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== 评估指标（除 LLM 外） ===")
    if isinstance(payload, list):
        for item in payload:
            print(f"{item['opponent']}: 胜率={item['winrate']} 平均回合={item['avg_game_length']}")
    else:
        print(payload)
    print(f"结果写入: {out_file}")


if __name__ == "__main__":
    main()
