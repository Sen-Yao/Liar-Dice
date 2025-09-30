"""神经网络结构模块

保持包级别导入轻量，避免在未安装依赖（torch、stable-baselines3）时出错。
如需使用，请直接从子模块导入：
    from rl_specialized.networks.policy_network import (
        MaskedStateFeatureExtractor, PolicyNetConfig, make_default_policy_kwargs
    )
"""

__all__ = [
    # 请直接从 policy_network 子模块导入所需对象
]
