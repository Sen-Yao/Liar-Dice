"""动作空间定义模块（专用动作空间）

示例：
    from rl_specialized.action_spaces import get_2_player_action_space
    asp = get_2_player_action_space()
"""

from .base import BaseActionSpace
from .player_specific import (
    PlayerSpecificActionSpace,
    create_player_specific_action_space,
    get_2_player_action_space,
    get_3_player_action_space,
    get_4_player_action_space,
    get_5_player_action_space,
    get_6_player_action_space,
)

__all__ = [
    'BaseActionSpace',
    'PlayerSpecificActionSpace',
    'create_player_specific_action_space',
    'get_2_player_action_space',
    'get_3_player_action_space',
    'get_4_player_action_space',
    'get_5_player_action_space',
    'get_6_player_action_space',
]
