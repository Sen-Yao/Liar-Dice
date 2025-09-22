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

### 动作映射规则

专用模型的动作空间采用简洁高效的ID映射：

```python
# 动作空间大小计算（以2人游戏为例）
max_dice = 2 * 5 = 10
total_actions = 1 + (2 * 10 * 6) = 121

# 动作ID分配：
- Challenge: action_id = 0
- 斋模式猜测: action_id = 1 + (count-1)*6 + (face-1)
- 飞模式猜测: action_id = 1 + 60 + (count-1)*6 + (face-1)
```

### 不同玩家数量的动作空间大小

| 玩家数量 | 最大骰子数 | 动作空间大小 | 内存占用 |
|---------|-----------|-------------|---------|
| 2人     | 10        | 121         | 最小    |
| 3人     | 15        | 181         | 小      |
| 4人     | 20        | 241         | 中等    |
| 5人     | 25        | 301         | 较大    |
| 6人     | 30        | 361         | 最大    |

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
- 支持批量编码和特征名称映射

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