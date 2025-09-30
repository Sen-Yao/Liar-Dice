import weakref
from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    """残差 MLP 模块：Linear -> SiLU -> Dropout -> Linear + 残差 -> LayerNorm
    适合数值型状态特征，稳定训练并提高表达能力
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.do = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.do(h)
        h = self.fc2(h)
        return self.ln(x + h)


class MaskedStateFeatureExtractor(BaseFeaturesExtractor):
    """带动作掩码注入的特征提取器

    - 输入：Dict(obs: Box, action_mask: MultiBinary)
    - 仅对 obs 做编码，掩码通过 policy_ref 注入到策略以屏蔽非法动作
    - 结构：LayerNorm 输入 + Linear(→hidden) + 2×ResidualBlock + Linear(→features_dim)
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, policy_ref: Optional[nn.Module] = None):
        assert isinstance(observation_space, gym.spaces.Dict)
        super().__init__(observation_space, features_dim)
        # Don't store policy reference to avoid serialization issues
        # Instead, we'll use a different approach for passing masks
        obs_space = observation_space["obs"]
        assert isinstance(obs_space, gym.spaces.Box)
        input_dim = int(obs_space.shape[0])

        hidden = max(64, features_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.inp = nn.Linear(input_dim, hidden)
        self.block1 = ResidualBlock(hidden, dropout=0.1)
        self.block2 = ResidualBlock(hidden, dropout=0.1)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, features_dim))

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # Store action mask for later use by policy
        if "action_mask" in observations:
            mask = observations["action_mask"].to(dtype=th.bool)
            # Store the mask as a module attribute for the policy to access
            self._current_action_mask = mask

        x = observations["obs"]
        x = self.norm(x)
        x = self.inp(x)
        x = nn.functional.silu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.out(x)
        return x


@dataclass
class PolicyNetConfig:
    """策略网络配置（供外部生成 policy_kwargs 使用）"""
    features_dim: int = 128
    pi_hidden: int = 128
    vf_hidden: int = 128


def make_default_policy_kwargs(cfg: Optional[PolicyNetConfig] = None):
    """生成与 SB3 兼容的 policy_kwargs：
    - 使用 MaskedStateFeatureExtractor 作为 features_extractor
    - 使用对称的 pi/vf MLP 结构
    """
    if cfg is None:
        cfg = PolicyNetConfig()
    return dict(
        features_extractor_class=MaskedStateFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=cfg.features_dim),
        net_arch=dict(
            pi=[cfg.pi_hidden, cfg.pi_hidden // 2],
            vf=[cfg.vf_hidden, cfg.vf_hidden // 2],
        ),
    )
