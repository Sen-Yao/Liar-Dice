#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_specialized 模块快速自检（单文件）

本脚本做以下检查并打印清晰结果：
- 包导入是否正常（避免包级别导入重依赖造成的崩溃）
- 动作空间尺寸是否符合 README 设计（2..6 人）
- 动作 ID ↔ 动作对象 映射是否双向一致
- 合法动作掩码（首轮不能 Challenge）是否与环境规则一致
- StateEncoder 维度与基本编码正确
- 单智能体训练包装环境（不依赖 SB3）是否能 reset/step 基本运行

运行方式：
    python rl_specialized_test_smoke.py

注意：
- 若未安装 gymnasium/pettingzoo，涉及环境的测试会被跳过并给出提示。
- 本测试不依赖 stable-baselines3/torch（训练相关会跳过）。
"""

from __future__ import annotations

import sys
import traceback

print("================ RL Specialized Smoke Test ================")

ok = 0
fail = 0


def section(title: str):
    print(f"\n--- {title} ---")


# 1) 包导入（避免包级别导入导致的崩溃）
section("Package Import")
deps_ok = True
try:
    import numpy as _np  # 许多组件依赖 numpy
except Exception as e:
    deps_ok = False
    print("[SKIP] Missing dependency (numpy):", e)

try:
    import rl_specialized
    if deps_ok:
        from rl_specialized.action_spaces import (
            get_2_player_action_space,
            get_3_player_action_space,
            get_4_player_action_space,
            get_5_player_action_space,
            get_6_player_action_space,
        )
        from rl_specialized.utils import create_state_encoder
    print(f"rl_specialized version: {getattr(rl_specialized, '__version__', 'unknown')}")
    ok += 1
except Exception as e:
    if deps_ok:
        print("[FAIL] Import error:", e)
        traceback.print_exc()
        fail += 1
    else:
        print("[SKIP] Package import skipped due to missing deps.")


# 2) 动作空间尺寸符合 README 设计
section("Action Space Size Check (2..6 players)")
expected_sizes = {
    2: 97,
    3: 145,
    4: 193,
    5: 241,
    6: 289,
}
try:
    if not deps_ok:
        raise ImportError("numpy not available")
    factories = {
        2: get_2_player_action_space,
        3: get_3_player_action_space,
        4: get_4_player_action_space,
        5: get_5_player_action_space,
        6: get_6_player_action_space,
    }
    for n in range(2, 7):
        asp = factories[n]()
        size = asp.get_action_space_size()
        print(f"num_players={n} -> action_space_size={size}")
        assert size == expected_sizes[n], f"size mismatch for {n} players: {size} != {expected_sizes[n]}"
    ok += 1
except ImportError as e:
    print("[SKIP] Action space size check skipped:", e)
except Exception as e:
    print("[FAIL] Action space size check:", e)
    traceback.print_exc()
    fail += 1


# 3) 映射一致性：id -> action -> id
section("Action ID <-> Object Round-trip")
try:
    if not deps_ok:
        raise ImportError("numpy not available")
    asp = get_2_player_action_space()
    total = asp.get_action_space_size()
    for aid in range(total):
        obj = asp.id_to_action(aid)
        back = asp.action_to_id(obj)
        if aid != back:
            raise AssertionError(f"round-trip mismatch at id={aid}: back={back}, obj={obj}")
    print(f"Checked {total} actions, all consistent.")
    ok += 1
except ImportError as e:
    print("[SKIP] Round-trip mapping skipped:", e)
except Exception as e:
    print("[FAIL] Round-trip mapping:", e)
    traceback.print_exc()
    fail += 1


# 4) 掩码正确性（首轮不能 Challenge），与环境规则一致
section("Mask Correctness vs Env (first turn)")
try:
    import env as game_env
    # 使用专用动作空间（min_count = n+1）
    e = game_env.LiarDiceEnv(num_players=2, dice_per_player=5, use_specialized_action_space=True)
    e.reset(seed=123)
    # 取 player_0 视角观测
    obs = e.observe("player_0")
    mask_env = e.get_action_mask(obs)

    # 使用我们的动作空间生成掩码
    asp = get_2_player_action_space()
    mask_asp = asp.get_action_mask(obs)

    # 首轮必须不能 challenge（id=0）
    assert mask_env[0] == False and mask_asp[0] == False, "first turn should not allow Challenge"
    # 与环境逐位一致
    if mask_env.shape != mask_asp.shape:
        raise AssertionError(f"mask shape mismatch: env={mask_env.shape}, asp={mask_asp.shape}")
    diff = (mask_env != mask_asp).nonzero()[0]
    assert diff.size == 0, f"mask differs at indices: {diff[:10]}"
    print("Mask consistent with env on first turn; challenge disabled as expected.")
    ok += 1
except ImportError as e:
    print("[SKIP] gymnasium/pettingzoo not available or env import failed:", e)
except Exception as e:
    print("[FAIL] Mask correctness vs env:", e)
    traceback.print_exc()
    fail += 1


# 5) StateEncoder 基本检查
section("StateEncoder Basic Check")
try:
    if not deps_ok:
        raise ImportError("numpy not available")
    encoder = create_state_encoder(num_players=2)
    # 构造一个最小可用观测（字段名需与 env 返回一致）
    sample_obs = {
        "my_dice_counts": (0, 1, 1, 1, 2, 0),
        "player_penalties": (0, 0),
        "last_guess": None,
        "total_dice_on_table": 10,
        "current_player_id_idx": 0,
        "is_my_turn": True,
    }
    vec = encoder.encode_observation(sample_obs)
    print("Encoded shape:", vec.shape, "feature_size:", encoder.get_feature_size())
    assert vec.shape == (encoder.get_feature_size(),)
    ok += 1
except ImportError as e:
    print("[SKIP] StateEncoder skipped:", e)
except Exception as e:
    print("[FAIL] StateEncoder:", e)
    traceback.print_exc()
    fail += 1


# 6) 单智能体包装环境 smoke（不依赖 SB3，仅 reset/step 若干步）
section("Single-Agent Wrapper Smoke Run")
try:
    import numpy as np
    from rl_specialized.training.env_wrappers import LiarDiceSingleAgentEnv

    env = LiarDiceSingleAgentEnv(num_players=2, dice_per_player=5)
    obs, info = env.reset(seed=42)
    steps = 0
    total_reward = 0.0
    while steps < 20:
        mask = obs["action_mask"].astype(bool)
        valid_ids = np.flatnonzero(mask)
        if valid_ids.size == 0:
            action = 0
        else:
            # 随机合法动作
            action = int(np.random.choice(valid_ids))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    print(f"Ran {steps} steps, total_reward={total_reward:.3f}")
    ok += 1
except ImportError as e:
    print("[SKIP] gymnasium not available for wrapper:", e)
except Exception as e:
    print("[FAIL] Single-Agent wrapper:", e)
    traceback.print_exc()
    fail += 1


print("\n================ Summary ================")
print(f"Passed: {ok}  Failed: {fail}")
print("=======================================")

sys.exit(0 if fail == 0 else 1)
