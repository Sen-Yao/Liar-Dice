from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from rl_specialized.networks.policy_network import MaskedStateFeatureExtractor


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """带动作掩码的策略：将非法动作的 logits 置为 -1e9
    依赖 MaskedFeatureExtractor 将 action_mask 注入到 self._last_action_mask
    """

    def __init__(self, *args, **kwargs):
        # 默认使用我们在 networks 中定义的特征提取器
        kwargs.setdefault("features_extractor_class", MaskedStateFeatureExtractor)
        fe_kwargs = kwargs.get("features_extractor_kwargs", {})
        fe_kwargs["policy_ref"] = self  # 让提取器将 mask 注入到本策略
        kwargs["features_extractor_kwargs"] = fe_kwargs
        super().__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Any:
        # 标准 logits
        action_logits = self.action_net(latent_pi)
        # 应用 mask（如可用）
        mask: Optional[th.Tensor] = getattr(self, "_last_action_mask", None)
        if mask is not None:
            # 广播到相同 batch 维度
            if mask.shape != action_logits.shape:
                # 可能 batch=1 的情况，尝试匹配
                if mask.dim() == 1 and action_logits.dim() == 2 and action_logits.shape[0] == 1:
                    mask = mask.unsqueeze(0)
            very_neg = th.finfo(action_logits.dtype).min / 2
            masked_logits = th.where(mask, action_logits, very_neg)
            return self.action_dist.proba_distribution(action_logits=masked_logits)
        return self.action_dist.proba_distribution(action_logits=action_logits)
