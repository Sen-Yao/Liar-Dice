# RL Specialized Models - 专用模型强化学习

## 📖 概述

这个模块实现了针对不同玩家数量（2-6人）的专用强化学习模型。与通用模型不同，专用模型为每种玩家配置训练独立的神经网络，以获得最优的性能表现。

## 🏗️ 架构设计

### 核心思想
- **专用优化**：每个玩家数量使用独立的动作空间和模型
- **高效训练**：动作空间精确匹配游戏配置，减少无效动作
- **性能最优**：针对特定场景深度优化，避免通用模型的复杂性

### 目录结构

```
rl_specialized/
├── action_spaces/          # 动作空间定义
│   ├── __init__.py
│   ├── base.py            # 动作空间基础抽象类
│   └── player_specific.py # 专用模型动作空间实现
├── agents/                # RL智能体实现
│   ├── __init__.py
│   └── specialized_agent.py (待实现)
├── networks/              # 神经网络结构
│   ├── __init__.py
│   └── policy_network.py (待实现)
├── training/              # 训练脚本
│   ├── __init__.py
│   └── train_specialized.py (待实现)
├── utils/                 # 工具函数
│   ├── __init__.py
│   └── state_encoder.py  # 状态编码器
└── README.md             # 本文件
```

## 🎯 动作空间设计

### 动作映射规则（专用）

专用模型对动作空间进行“首轮规则化”裁剪：计数从 n+1 开始（n=玩家数）。

```python
# 以2人游戏为例
num_players = 2
dice_per_player = 5
min_count = n + 1 = 3
max_count = n * 5 = 10
counts_per_mode = max_count - min_count + 1 = 8

total_actions = 1 (challenge) + 2 * counts_per_mode * 6 = 97

# 动作ID分配：
- Challenge: action_id = 0
- 斋模式猜测: action_id = 1 + (count-min_count)*6 + (face-1)
- 飞模式猜测: action_id = 1 + counts_per_mode*6 + (count-min_count)*6 + (face-1)
```

### 不同玩家数量的动作空间大小

| 玩家数量 | 最大骰子数 | 计数范围（含） | 动作空间大小 |
|---------|-----------|---------------|-------------|
| 2人     | 10        | 3..10         | 97          |
| 3人     | 15        | 4..15         | 145         |
| 4人     | 20        | 5..20         | 193         |
| 5人     | 25        | 6..25         | 241         |
| 6人     | 30        | 7..30         | 289         |

## 🧠 核心组件

### 1. BaseActionSpace (base.py)
- 抽象基类，定义动作空间的通用接口
- 实现游戏规则验证逻辑
- 提供跨模式比较算法

### 2. PlayerSpecificActionSpace (player_specific.py)
- 专用模型的具体实现
- 高效的动作ID与对象互转
- 合法动作掩码生成
- 预定义2-6人游戏配置

### 3. StateEncoder (state_encoder.py)
- 将复杂游戏状态编码为神经网络输入
- 固定长度特征向量：手牌(6) + 玩家信息(n) + 猜测(4) + 游戏状态(3)
- 兼容环境返回的 Guess（dataclass）与 dict 的 last_guess
- 支持批量编码和特征名称映射

### 4. Policy Network (networks/policy_network.py)
- 面向数值型状态的高效 MLP + 残差结构
- 架构：LayerNorm → Linear → 2×ResidualBlock(Linear+SiLU+Dropout→Linear+残差+LayerNorm) → Linear
- 默认输出特征维度 `features_dim=128`，便于策略/价值头共享
- 提供 `MaskedStateFeatureExtractor` 用于与 SB3 集成（从观察中注入动作掩码）
- 提供 `make_default_policy_kwargs()` 辅助函数，用于生成 SB3 的 `policy_kwargs`

## 🚀 快速开始

### 创建2人游戏动作空间

```python
from rl_specialized.action_spaces import get_2_player_action_space

# 创建专用动作空间
action_space = get_2_player_action_space()

# 获取动作空间信息
print(f"动作空间大小: {action_space.get_action_space_size()}")
print(f"分布信息: {action_space.get_action_distribution_info()}")

# 动作转换
action_id = 5
action_obj = action_space.id_to_action(action_id)
print(f"动作ID {action_id} -> {action_obj}")

# 获取合法动作掩码
observation = {...}  # 来自环境的观察
mask = action_space.get_action_mask(observation)
valid_actions = action_space.get_valid_actions(observation)
```

### 状态编码

```python
from rl_specialized.utils import create_state_encoder

# 创建状态编码器
encoder = create_state_encoder(num_players=2)

# 编码单个观察
observation = {...}  # 来自环境
encoded_state = encoder.encode_observation(observation)
print(f"编码后状态维度: {encoded_state.shape}")

# 批量编码
observations = [obs1, obs2, obs3]
batch_states = encoder.encode_batch(observations)
```

## 📊 性能特点

### 优势
- **内存效率**：动作空间针对特定玩家数量最小化
- **训练速度**：减少无效动作探索，加快收敛
- **模型精度**：专门优化，避免通用模型的性能妥协
- **可解释性**：每个模型对应明确的游戏配置

### 适用场景
- 固定玩家数量的游戏环境
- 对性能要求极高的生产环境
- 需要详细分析特定配置的研究场景
- 资源充足，可以维护多个模型的情况

## 🔧 开发指南

### 扩展新玩家数量

如需支持7人或更多玩家：

1. 在 `player_specific.py` 中添加新的工厂函数
2. 更新 `__init__.py` 导出新函数
3. 验证动作空间大小和内存占用

### 自定义状态编码

StateEncoder支持继承和自定义：

```python
class CustomStateEncoder(StateEncoder):
    def encode_observation(self, observation):
        # 添加自定义特征
        base_features = super().encode_observation(observation)
        custom_features = self._extract_custom_features(observation)
        return np.concatenate([base_features, custom_features])
```

## 🧪 测试与验证

### 动作空间验证

```python
# 验证动作映射的双向一致性
action_space = get_2_player_action_space()
for action_id in range(action_space.get_action_space_size()):
    action_obj = action_space.id_to_action(action_id)
    recovered_id = action_space.action_to_id(action_obj)
    assert action_id == recovered_id
```

### 掩码正确性测试

```python
# 测试合法动作掩码
observation = create_test_observation()
mask = action_space.get_action_mask(observation)
for action_id in range(len(mask)):
    if mask[action_id]:
        action_obj = action_space.id_to_action(action_id)
        assert is_legal_in_game_context(action_obj, observation)
```

## 📈 下一步计划

1. **实现SpecializedAgent类** - 基于专用动作空间的RL智能体
2. **设计PolicyNetwork** - 针对不同玩家数量优化的网络结构
3. **开发训练脚本** - 支持并行训练多个专用模型
4. **性能对比实验** - 与通用模型的详细性能对比
5. **模型融合策略** - 探索多模型集成的可能性

## 🤝 贡献指南

- 遵循项目编码规范：英文命名 + 中文注释
- 保持代码简洁明了，避免过度设计
- 添加充分的单元测试和文档
- 性能优化时保持代码可读性

---

*最后更新: 2024年9月22日*
*版本: v1.0.0*
*作者: 骰子骗子RL项目组*
### 训练（Torch + SB3）

```bash
pip install "stable-baselines3[extra]" torch gymnasium tensorboard

# 训练（自动选择 cuda/mps/cpu）
python -m rl_specialized.training.train_specialized --num_players 2 --timesteps 200000

# TensorBoard
tensorboard --logdir runs/rl_specialized
```

内部集成：
- 单智能体 Gym 包装（RL 控制 player_0，其余为 BasicRuleAgent）
- 自定义策略 Mask：在 logits 处屏蔽非法动作（动作掩码）
- 特征提取器采用 `MaskedStateFeatureExtractor`，默认 `features_dim=128`

如需自定义网络宽度，可在 `train_specialized.py` 中调整：

```python
from rl_specialized.networks.policy_network import make_default_policy_kwargs, PolicyNetConfig

policy_kwargs = make_default_policy_kwargs(PolicyNetConfig(features_dim=128, pi_hidden=128, vf_hidden=128))
model = PPO(MaskedActorCriticPolicy, env, policy_kwargs=policy_kwargs, ...)
```

### 自博弈训练（混合对手池，简化实现）

- 思路：RL 控制 player_0，对手来自“对手池”（多参数规则体 + 历史策略快照）。训练中按进度线性降低规则体占比，定期将当前策略快照加入对手池。
- 文件：`rl_specialized/training/env_wrappers.py`（自博弈环境与对手池）+ `rl_specialized/training/train_selfplay.py`（训练脚本）。

运行：

```bash
python -m rl_specialized.training.train_selfplay --num_players 2 --timesteps 2000000 --snapshot_freq 200000
```

要点：
- 对手池初始包含多参数规则对手（起手面值∈{3,4,5}，挑战阈值偏移∈{2,3,4,5}）。
- 规则体占比从 0.8 → 0.2（线性随训练进度下降）。
- 每 `snapshot_freq` 步将当前策略保存并加入对手池（推断在 CPU 上进行，不占用训练设备显存）。
- 观测与动作掩码与专用训练一致：`{'obs': state_vec, 'action_mask': mask}`。

#### 奖励潜在塑形（默认开启）

- 目的：在不改变最优策略的前提下，提供更密集的学习信号，提升稳定性与收敛速度。
- 形式：在自博弈环境中加入势函数塑形项

  `F = β · (γ · Φ(s') − Φ(s))`

  其中 Φ(s) 估计当前最后叫点相对于“可信度”的差距：
  - 飞：成功概率 p(face 或 1) = 2/6；斋：p(face) = 1/6
  - 期望总成功 E = 自己手牌成功数 + 未知骰子数 × p
  - Φ(s) = clip((E − 最后叫点count)/总骰子数, −1, 1)

- 参数：在 `train_selfplay.py` 中默认 `β=0.05`，`γ` 与 PPO 的 `gamma` 保持一致（默认 0.99）。
- 关闭方式：如需关闭，在创建 `LiarDiceSelfPlayEnv` 时将 `dense_shaping=False`。
