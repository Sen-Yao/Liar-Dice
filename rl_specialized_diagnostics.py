#!/usr/bin/env python3
"""Diagnostic script for the `rl_specialized` package.

The script exercises the core building blocks implemented under `rl_specialized/` and
prints human-readable results so you can quickly verify whether the specialization
layer behaves as described in `rl_specialized/README.md`.

Each check is self-contained. If optional dependencies (gymnasium, pettingzoo,
stable-baselines3, torch) are missing, the corresponding section is skipped but the
rest of the diagnostics continue to run.
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass
from importlib import util as importlib_util
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class CheckResult:
    name: str
    passed: bool
    skipped: bool = False
    detail: str = ""

    def format_line(self) -> str:
        if self.skipped:
            return f"[SKIP] {self.name}: {self.detail}"
        prefix = "[ OK ]" if self.passed else "[FAIL]"
        return f"{prefix} {self.name}: {self.detail}"


def run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name=name, passed=True, detail=detail)
    except SkipCheck as exc:
        return CheckResult(name=name, passed=False, skipped=True, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - we want full diagnostics
        formatted = f"{exc}\n" + traceback.format_exc()
        return CheckResult(name=name, passed=False, detail=formatted.strip())


class SkipCheck(RuntimeError):
    """Raised when a check should be skipped (missing optional dependency, etc.)."""


def require_module(module_name: str) -> Any:
    try:
        __import__(module_name)
        return sys.modules[module_name]
    except Exception as exc:  # noqa: BLE001 - propagate message for diagnostics
        raise SkipCheck(f"module `{module_name}` is not available: {exc}") from exc


def ensure_env_stack() -> None:
    """Verify that the environment dependencies exist before importing heavy modules."""

    missing: List[str] = []
    for dep in ("gymnasium", "pettingzoo"):
        if importlib_util.find_spec(dep) is None:
            missing.append(dep)
    if missing:
        raise SkipCheck("missing dependencies: " + ", ".join(missing))


def check_imports() -> str:
    numpy = require_module("numpy")
    rl_specialized = require_module("rl_specialized")

    ensure_env_stack()

    from rl_specialized.action_spaces import (
        get_2_player_action_space,
        get_3_player_action_space,
        get_4_player_action_space,
        get_5_player_action_space,
        get_6_player_action_space,
    )
    from rl_specialized.utils import create_state_encoder
    from rl_specialized.agents import SpecializedAgent

    details = [
        f"numpy {numpy.__version__}",
        f"rl_specialized {getattr(rl_specialized, '__version__', 'unknown')}",
        f"exports: {[fn.__name__ for fn in (get_2_player_action_space, get_3_player_action_space, get_4_player_action_space, get_5_player_action_space, get_6_player_action_space, create_state_encoder, SpecializedAgent)]}",
    ]
    return "; ".join(details)


def check_action_space_sizes() -> str:
    ensure_env_stack()

    from rl_specialized.action_spaces import (
        get_2_player_action_space,
        get_3_player_action_space,
        get_4_player_action_space,
        get_5_player_action_space,
        get_6_player_action_space,
    )

    expected = {2: 97, 3: 145, 4: 193, 5: 241, 6: 289}
    mismatches: List[str] = []

    for players, factory in {
        2: get_2_player_action_space,
        3: get_3_player_action_space,
        4: get_4_player_action_space,
        5: get_5_player_action_space,
        6: get_6_player_action_space,
    }.items():
        space = factory()
        size = space.get_action_space_size()
        if size != expected[players]:
            mismatches.append(f"{players}p => {size} (expected {expected[players]})")

    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    return "sizes match README for 2-6 players"


def check_round_trip() -> str:
    ensure_env_stack()

    from rl_specialized.action_spaces import get_2_player_action_space

    space = get_2_player_action_space()
    failures: List[str] = []
    for action_id in range(space.get_action_space_size()):
        action_obj = space.id_to_action(action_id)
        recovered = space.action_to_id(action_obj)
        if action_id != recovered:
            failures.append(f"id {action_id} -> {action_obj} -> {recovered}")
            if len(failures) >= 5:
                break
    if failures:
        raise RuntimeError("round-trip mismatches: " + "; ".join(failures))
    return f"validated {space.get_action_space_size()} discrete actions"


def check_mask_alignment() -> str:
    ensure_env_stack()

    env_module = require_module("env")
    from rl_specialized.action_spaces import get_3_player_action_space

    if not hasattr(env_module, "LiarDiceEnv"):
        raise SkipCheck("`env` module does not expose LiarDiceEnv")

    game_env = env_module.LiarDiceEnv(num_players=3, dice_per_player=5, use_specialized_action_space=True)
    game_env.reset(seed=123)
    obs = game_env.observe("player_0")
    mask_env = game_env.get_action_mask(obs)

    space = get_3_player_action_space()
    mask_space = space.get_action_mask(obs)

    if mask_env.shape != mask_space.shape:
        raise RuntimeError(f"mask shape mismatch: env {mask_env.shape} vs space {mask_space.shape}")
    diff = (mask_env != mask_space).nonzero()[0]
    if diff.size:
        indices = ", ".join(str(int(idx)) for idx in diff[:10])
        raise RuntimeError(f"mask differs at indices: {indices}")

    if mask_env[0]:
        raise RuntimeError("challenge became legal on the first turn; expected False")
    return "mask identical to environment on first turn"


def check_state_encoder() -> str:
    ensure_env_stack()

    from rl_specialized.utils import create_state_encoder

    encoder = create_state_encoder(num_players=4)
    sample = {
        "my_dice_counts": (1, 0, 2, 1, 0, 1),
        "player_penalties": (0, 1, 0, 2),
        "last_guess": None,
        "total_dice_on_table": 20,
        "current_player_id_idx": 1,
        "is_my_turn": False,
    }
    vec = encoder.encode_observation(sample)
    if vec.shape != (encoder.get_feature_size(),):
        raise RuntimeError(f"unexpected feature shape: {vec.shape}")
    finite = bool(math.isfinite(float(vec.sum())))
    if not finite:
        raise RuntimeError("encoder produced non-finite values")
    return f"feature_dim={encoder.get_feature_size()} sample_sum={float(vec.sum()):.3f}"


def check_specialized_agent_policy() -> str:
    ensure_env_stack()

    from rl_specialized.agents import SpecializedAgent

    agent = SpecializedAgent(num_players=2)
    dummy_obs = {
        "my_dice_counts": (1, 1, 1, 1, 1, 0),
        "player_penalties": (0, 0),
        "last_guess": None,
        "total_dice_on_table": 10,
        "current_player_id_idx": 0,
        "is_my_turn": True,
    }
    action_id = agent.select_action_id(dummy_obs)
    mask = agent.action_space.get_action_mask(dummy_obs)
    if not mask[action_id]:
        raise RuntimeError(f"agent sampled illegal action {action_id}")
    obj = agent.id_to_action(action_id)
    return f"sampled action id={action_id}, object={obj}"


def check_single_agent_env_rollout() -> str:
    ensure_env_stack()

    from rl_specialized.training.env_wrappers import LiarDiceSingleAgentEnv

    env = LiarDiceSingleAgentEnv(num_players=2, dice_per_player=5)
    obs, _ = env.reset(seed=7)
    total_reward = 0.0
    for step in range(25):
        mask = obs["action_mask"].astype(bool)
        valid_ids = mask.nonzero()[0]
        action = int(valid_ids[0]) if valid_ids.size else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    return f"rolled {step + 1} steps, accumulated reward {total_reward:.3f}"


def main() -> None:
    checks: List[Tuple[str, Callable[[], str]]] = [
        ("Imports", check_imports),
        ("Action space sizes", check_action_space_sizes),
        ("Action ID round-trip", check_round_trip),
        ("Mask alignment with env", check_mask_alignment),
        ("State encoder", check_state_encoder),
        ("SpecializedAgent sampling", check_specialized_agent_policy),
        ("Single-agent wrapper rollout", check_single_agent_env_rollout),
    ]

    results: List[CheckResult] = [run_check(name, fn) for name, fn in checks]

    print("================ rl_specialized diagnostics ================")
    for result in results:
        print(result.format_line())
    passed = sum(result.passed for result in results)
    skipped = sum(result.skipped for result in results)
    failed = len(results) - passed - skipped
    print("------------------------------------------------------------")
    print(f"Summary: passed={passed} failed={failed} skipped={skipped}")
    print("============================================================")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
