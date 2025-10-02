#!/usr/bin/env python3
"""验证飞模式禁止喊1的修改是否生效"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env import LiarDiceEnv, Guess
from rl_specialized.action_spaces.player_specific import PlayerSpecificActionSpace


def test_env_is_legal():
    """测试 env.py 的 _is_legal() 方法"""
    print("=" * 60)
    print("测试 1: env.py 的 _is_legal() 方法")
    print("=" * 60)

    env = LiarDiceEnv(num_players=2, dice_per_player=5, use_specialized_action_space=True)
    env.reset()

    # 测试飞模式喊1（应该非法）
    fei_1_guess = Guess(mode='飞', count=3, face=1)
    is_legal = env._is_legal(fei_1_guess)

    print(f"飞模式喊1 (count=3, face=1): {'合法' if is_legal else '非法'} ✓" if not is_legal else f"飞模式喊1 (count=3, face=1): {'合法' if is_legal else '非法'} ✗")

    # 测试飞模式喊其他面值（应该合法）
    fei_2_guess = Guess(mode='飞', count=3, face=2)
    is_legal = env._is_legal(fei_2_guess)

    print(f"飞模式喊2 (count=3, face=2): {'合法' if is_legal else '非法'} ✓" if is_legal else f"飞模式喊2 (count=3, face=2): {'合法' if is_legal else '非法'} ✗")

    # 测试斋模式喊1（应该合法）
    zhai_1_guess = Guess(mode='斋', count=3, face=1)
    is_legal = env._is_legal(zhai_1_guess)

    print(f"斋模式喊1 (count=3, face=1): {'合法' if is_legal else '非法'} ✓" if is_legal else f"斋模式喊1 (count=3, face=1): {'合法' if is_legal else '非法'} ✗")

    print()


def test_action_space_mask():
    """测试 PlayerSpecificActionSpace 的动作掩码"""
    print("=" * 60)
    print("测试 2: PlayerSpecificActionSpace 的动作掩码")
    print("=" * 60)

    action_space = PlayerSpecificActionSpace(num_players=2, dice_per_player=5)

    # 首轮观测
    obs = {"last_guess": None}
    mask = action_space.get_action_mask(obs)

    # 检查所有动作
    fei_1_actions = []
    fei_other_actions = []
    zhai_1_actions = []

    for action_id in range(action_space.get_action_space_size()):
        action = action_space.id_to_action(action_id)
        if isinstance(action, Guess):
            if action.mode == '飞' and action.face == 1:
                fei_1_actions.append((action_id, mask[action_id]))
            elif action.mode == '飞' and action.face != 1:
                fei_other_actions.append((action_id, mask[action_id]))
            elif action.mode == '斋' and action.face == 1:
                zhai_1_actions.append((action_id, mask[action_id]))

    # 验证飞喊1被屏蔽
    all_fei_1_blocked = all(not masked for _, masked in fei_1_actions)
    print(f"飞模式 face=1 的动作数量: {len(fei_1_actions)}")
    print(f"飞模式 face=1 全部被屏蔽: {'是 ✓' if all_fei_1_blocked else '否 ✗'}")

    # 验证飞喊其他面值合法
    some_fei_other_legal = any(masked for _, masked in fei_other_actions[:5])
    print(f"飞模式 face∈{2,3,4,5,6} 部分合法: {'是 ✓' if some_fei_other_legal else '否 ✗'}")

    # 验证斋喊1合法
    some_zhai_1_legal = any(masked for _, masked in zhai_1_actions)
    print(f"斋模式 face=1 部分合法: {'是 ✓' if some_zhai_1_legal else '否 ✗'}")

    print(f"\n动作空间总大小: {action_space.get_action_space_size()}")
    print(f"首轮合法动作数: {mask.sum()}")
    print()


def test_env_action_mask():
    """测试 env 的 get_action_mask()"""
    print("=" * 60)
    print("测试 3: env.get_action_mask() 方法")
    print("=" * 60)

    env = LiarDiceEnv(num_players=2, dice_per_player=5, use_specialized_action_space=True)
    env.reset()

    obs = env.observe(env.agent_selection)
    mask = env.get_action_mask(obs)

    # 检查飞喊1的动作ID
    fei_1_blocked_count = 0
    fei_1_total_count = 0

    for action_id in range(len(mask)):
        action = env.action_to_object(action_id)
        if isinstance(action, Guess) and action.mode == '飞' and action.face == 1:
            fei_1_total_count += 1
            if not mask[action_id]:
                fei_1_blocked_count += 1

    print(f"飞模式 face=1 的动作总数: {fei_1_total_count}")
    print(f"飞模式 face=1 被屏蔽数量: {fei_1_blocked_count}")
    print(f"飞模式 face=1 全部屏蔽: {'是 ✓' if fei_1_blocked_count == fei_1_total_count else '否 ✗'}")

    print(f"\nenv 动作空间总大小: {len(mask)}")
    print(f"首轮合法动作数: {mask.sum()}")
    print()


if __name__ == "__main__":
    print("\n🔍 开始验证飞模式禁止喊1的修改\n")

    try:
        test_env_is_legal()
        test_action_space_mask()
        test_env_action_mask()

        print("=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
