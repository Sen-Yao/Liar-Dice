#!/usr/bin/env python3
"""单元测试：验证历史填充位置修复的正确性"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rl_specialized.utils.state_encoder import StateEncoder
from env import Guess


def test_history_padding_position():
    """测试历史填充位置是否正确（填充在前）"""
    print("=" * 60)
    print("测试：历史填充位置")
    print("=" * 60)

    encoder = StateEncoder(num_players=2, dice_per_player=5, history_length=3)

    # 基础观测（不包含历史）
    base_obs = {
        "my_dice_counts": [1, 1, 1, 1, 1, 0],
        "player_penalties": [0, 0],
        "last_guess": None,
        "total_dice_on_table": 10,
        "current_player_id_idx": 0,
        "is_my_turn": True,
    }

    # 测试1：无历史（首轮）
    print("\n测试1：无历史（首轮）")
    obs_no_history = {**base_obs, "game_round_history": []}
    state = encoder.encode_observation(obs_no_history)

    # 状态维度应为 6 + 2 + 4 + 3 + 12 = 27
    assert len(state) == 27, f"状态维度错误: {len(state)} != 27"

    # 历史部分应全为0（索引15-26，共12维）
    history_part = state[15:]
    assert np.allclose(history_part, 0.0), f"无历史时，历史部分应全为0，实际: {history_part}"
    print("✓ 无历史时，历史部分全为0")

    # 测试2：1步历史
    print("\n测试2：1步历史")
    obs_1_history = {
        **base_obs,
        "game_round_history": [(0, Guess('飞', 3, 4))]
    }
    state = encoder.encode_observation(obs_1_history)

    # 历史部分的前8维应为填充（0），后4维应为实际历史
    history_part = state[15:]
    padding_part = history_part[:8]
    actual_history = history_part[8:]

    assert np.allclose(padding_part, 0.0), f"填充部分应为0，实际: {padding_part}"
    assert actual_history.sum() > 0, f"实际历史部分应非零，实际: {actual_history}"

    # 验证最近历史在固定位置（最后4维）
    expected_features = [
        0.0,  # mode: 飞=0
        3.0 / 10.0,  # count normalized
        4.0 / 6.0,  # face normalized
        0.0 / 2.0,  # player_idx normalized
    ]
    assert np.allclose(actual_history, expected_features, atol=1e-6), \
        f"历史特征不匹配，期望: {expected_features}, 实际: {actual_history}"
    print(f"✓ 1步历史时，填充在前（前8维=0），实际历史在后（后4维非零）")
    print(f"  实际历史特征: {actual_history}")

    # 测试3：3步历史（无填充）
    print("\n测试3：3步历史（无填充）")
    obs_3_history = {
        **base_obs,
        "game_round_history": [
            (0, Guess('飞', 3, 4)),
            (1, Guess('飞', 3, 5)),
            (0, Guess('斋', 2, 6)),
        ]
    }
    state = encoder.encode_observation(obs_3_history)

    history_part = state[15:]
    # 所有12维都应该是非零历史（无填充）
    assert history_part.sum() > 0, f"3步历史时，历史部分应全为非零"

    # 验证最近一步（第3步）在最后4维
    step3_features = history_part[8:]
    expected_step3 = [
        1.0,  # mode: 斋=1
        2.0 / 10.0,  # count
        6.0 / 6.0,  # face
        0.0 / 2.0,  # player
    ]
    assert np.allclose(step3_features, expected_step3, atol=1e-6), \
        f"第3步特征不匹配，期望: {expected_step3}, 实际: {step3_features}"
    print(f"✓ 3步历史时，无填充，最近历史在末尾")
    print(f"  最近一步特征: {step3_features}")

    # 测试4：使用 game_round_history_encoded（类型安全版本）
    print("\n测试4：使用 game_round_history_encoded")
    obs_encoded = {
        **base_obs,
        "game_round_history_encoded": [
            {"mode": 0, "count": 3, "face": 4, "player_idx": 0},
        ]
    }
    state = encoder.encode_observation(obs_encoded)

    history_part = state[15:]
    padding_part = history_part[:8]
    actual_history = history_part[8:]

    assert np.allclose(padding_part, 0.0), "编码版本：填充部分应为0"
    assert actual_history.sum() > 0, "编码版本：实际历史应非零"
    print("✓ game_round_history_encoded 编码正确")

    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


def test_backward_compatibility():
    """测试向后兼容性（history_length=0）"""
    print("\n" + "=" * 60)
    print("测试：向后兼容性（history_length=0）")
    print("=" * 60)

    encoder = StateEncoder(num_players=2, dice_per_player=5, history_length=0)

    obs = {
        "my_dice_counts": [1, 1, 1, 1, 1, 0],
        "player_penalties": [0, 0],
        "last_guess": Guess('飞', 3, 4),
        "total_dice_on_table": 10,
        "current_player_id_idx": 0,
        "is_my_turn": True,
        "game_round_history": [(0, Guess('飞', 3, 4))],
    }

    state = encoder.encode_observation(obs)

    # 应该是旧维度：6 + 2 + 4 + 3 = 15
    assert len(state) == 15, f"history_length=0 时维度应为15，实际: {len(state)}"
    print(f"✓ history_length=0 时，维度保持为15（旧模型兼容）")

    print("=" * 60)


if __name__ == "__main__":
    try:
        test_history_padding_position()
        test_backward_compatibility()
        print("\n✅ 所有测试通过！历史填充位置修复正确。")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
