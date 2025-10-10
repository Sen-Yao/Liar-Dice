# Liar Dice 强化学习项目

这是一个基于深度强化学习的骗子骰子（Liar's Dice）智能体训练与对战平台。项目使用 PettingZoo 框架构建多智能体环境，并实现了基于 DQN 的训练算法。

## 目录

- [游戏规则](#游戏规则)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [训练与评估](#训练与评估)
- [技术特性](#技术特性)
- [算法说明](#算法说明)

---

## 游戏规则

### 基本玩法

骗子骰子是一个回合制多人游戏（2人及以上）。每位玩家拥有 5 颗六面骰子（点数 1-6），骰子结果**仅自己可见**。

**游戏流程**：
1. 所有玩家掷骰子
2. 按顺序轮流行动，每回合玩家必须选择：
   - **叫点**：宣称"场上所有玩家的骰子中，某个点数至少有 X 个"
   - **开牌**（Challenge）：质疑上一位玩家的叫点，翻开所有骰子验证真假

3. 开牌结算：
   - 若实际数量 ≥ 叫点数量 → 叫点者胜，开牌者输
   - 若实际数量 < 叫点数量 → 开牌者胜，叫点者输

**特殊规则**：
- 首轮玩家第一次叫点的数量必须 **> 玩家人数**（如 5 人局必须从 "6 个 X" 起叫）
- 每次叫点必须**严格大于**上一次叫点

### 飞与斋模式

本项目实现了进阶规则"飞"和"斋"：

#### 飞模式（正常模式）
- **点数 1 为万能牌（wild）**：计数时包括目标点数和所有 1
- **禁止叫 1**：不能选择点数 1 作为叫点目标
- **数字大小**：2 < 3 < 4 < 5 < 6（1 最小，因为是万能牌）
- **示例**：叫"7 个 4"意味着场上所有 4 和所有 1 加起来 ≥ 7 个

#### 斋模式
- **点数 1 就是 1**：不作为万能牌，只计算目标点数
- **允许叫 1**：可以叫"X 个 1"
- **数字大小**：2 < 3 < 4 < 5 < 6 < 1（1 最大，因为稀有）
- **示例**：叫"3 个 4"意味着场上只计算 4 的数量，不包括 1

#### 模式切换规则

玩家可以在飞和斋之间自由切换，但需满足以下条件：

| 当前模式 | 目标模式 | 切换条件 |
|---------|---------|---------|
| 飞 | 飞 | 数量更多，或数量相同时点数更大 |
| 斋 | 斋 | 数量更多，或数量相同时按"2<3<4<5<6<1"比较 |
| 飞 | 斋 | 新数量 ≥ ⌈旧数量 / 2⌉（向上取整），忽略点数 |
| 斋 | 飞 | 新数量 ≥ 旧数量 × 2，忽略点数 |

**权威实现**：所有规则比较逻辑统一封装在 `env.LiarDiceEnv._is_strictly_greater` 中。

---

## 快速开始

### 环境要求

```bash
# 安装 PyTorch（CUDA 12.4）
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# 或 CPU 版本
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch

# 安装其他依赖
pip install numpy gymnasium pettingzoo

# 如果使用 rl_specialized 训练方法，额外安装：
pip install "stable-baselines3[extra]" tensorboard
```

### 运行示例

```bash
# 训练 DQN 智能体（默认 10000 局）
python train.py

# 自定义训练参数
python train.py --num_episodes 5000 --learning_rate 0.0001 --n_step 3

# 人机对战（1 人类玩家 + 1 AI）
python main.py --mode human

# 多人局（1 人类 + 3 AI）
python main.py --mode human --num_players 4

# 加载模型进行评估
python model_playground.py --model-path models/dqn_model.pth --eval-episodes 100

# 人类 vs 训练好的模型
python model_playground.py --model-path models/dqn_model.pth --interactive

# 全面评估模型性能（对抗多种baseline）
python evaluator.py --model-path models/dqn_model.pth --num-games 1000 --save-results
```

---

## 项目结构

```
Liar-Dice/
├── env.py                    # PettingZoo 游戏环境实现（支持 history_len 参数）
├── agents/
│   ├── DQN_agent.py         # DQN 智能体（22维状态，组合动作Q）
│   ├── heuristic_agent.py   # 启发式规则智能体（含概率型变体）
│   ├── basic_agent.py       # 基础规则智能体（BasicRuleAgent, ProbabilisticBasicAgent）
│   ├── human_agent.py       # 人类玩家接口
│   ├── llm_agent.py         # LLM 智能体（基于通义千问API，三层合法性验证）
│   └── baseline_agents.py   # Baseline智能体（RandomAgent等，用于性能对比实验）
├── train/
│   └── DQN_train.py         # DQN 训练流程（DDQN + n-step + 奖励整形 + 对手类型选择）
├── train.py                 # 训练入口脚本（DQN 方法，自动保存 config.json）
├── main.py                  # 人机对战入口
├── model_playground.py      # 模型评估与调试工具
├── evaluator.py             # DQN模型全面评估工具（支持 --model-dir 和 --json 输出）
├── run_dqn_ablations.py     # DQN 消融实验自动化工具
├── utils.py                 # 工具函数（动作合法性检查等）
├── models/                  # 保存的模型权重（按 exp_tag 组织，包含 model.pth 和 config.json）
├── runs/ablations_dqn/      # 消融实验结果存储目录
└── rl_specialized/          # 专用 RL 模型（PPO + SB3，独立训练路线）
    ├── action_spaces/       # 玩家数量特定的动作空间
    ├── agents/              # 专用智能体实现
    ├── networks/            # 策略网络架构（带掩码支持）
    ├── training/            # 训练脚本（自博弈 + 对抗规则体）
    ├── utils/               # 状态编码器
    └── README.md            # 专用模型详细文档
```

---

## 训练与评估

本项目提供 **两种独立的训练方法**：

### 方法 1: DQN 训练（主要方法）

```bash
python train.py \
  --num_episodes 10000 \
  --learning_rate 0.0001 \
  --n_step 3 \
  --shaping_beta 0.2 \
  --target_soft_tau 0.005 \
  --min_epsilon 0.1 \
  --save_model_freq 1000 \
  --opponent_type mixed
```

**关键超参数**：
- `--n_step`：n-step 回报的步长（默认 3，加速稀疏奖励传播）
- `--shaping_beta`：势能奖励整形系数（默认 0.2）
- `--reward_shaping`：是否启用势能塑形（默认 True）
- `--ddqn`：是否启用 Double DQN（默认 True）
- `--target_soft_tau`：目标网络 Polyak 软更新系数（默认 0.005）
- `--min_epsilon`：探索率下限（默认 0.1，避免过早收敛）
- `--opponent_type`：对手类型（mixed/heuristic/selfplay，默认 mixed）
- `--heuristic_ratio`：启发式对手占比（默认 0.5，仅在 mixed 模式生效）
- `--history_len`：观测中保留的历史长度（0 表示不裁剪，默认 0）
- `--exp_tag`：实验标签（留空自动生成）
- `--quick_eval`：是否启用训练中快速评估（默认 False）

**DQN 模型评估**：

```bash
# 快速评估（100 局）
python model_playground.py \
  --model-path models/dqn_model.pth \
  --eval-episodes 100 \
  --num_players 4

# 人机交互对战
python model_playground.py \
  --model-path models/dqn_model.pth \
  --interactive
```

**全面评估工具（evaluator.py）**：

```bash
# 完整评估（对抗所有baseline，每种1000局）
python evaluator.py \
  --model-path models/dqn_model.pth \
  --num-games 1000 \
  --save-results

# 自定义评估（不包含启发式对手）
python evaluator.py \
  --model-path models/dqn_model.pth \
  --num-games 500 \
  --no-heuristic

# 包含LLM对手评估（需要API密钥）
python evaluator.py \
  --model-path models/dqn_model.pth \
  --num-games 1000 \
  --include-llm
```

**评估指标**：
- 胜率（vs Random、Conservative、Aggressive、Heuristic、LLM）
- 平均回合数
- 动作分布（猜测率、挑战率）
- 决策质量（非法动作率、fallback率）

### DQN 消融实验工具

项目提供 `run_dqn_ablations.py` 用于自动化运行消融实验：

```bash
# 运行单个消融实验
python run_dqn_ablations.py --ablation ddqn_off --num-episodes 1000 --eval-games 300

# 可用的消融实验：
# - baseline: 默认配置（DDQN + 势能塑形 + n-step=3 + mixed对手）
# - ddqn_off: 关闭 Double DQN
# - reward_shaping_off: 关闭势能奖励塑形
# - n_step_1: 使用 1-step TD（关闭 n-step）
# - history_3: 历史长度限制为 3
# - history_5: 历史长度限制为 5
# - opponent_heuristic: 纯启发式对手
# - opponent_selfplay: 纯自博弈对手
```

**输出**：
- 训练好的模型保存在 `models/<exp_tag>/`
- 评估结果保存在 `runs/ablations_dqn/<exp_tag>/results.json`

---

### 方法 2: 专用 PPO 训练（rl_specialized）

针对特定玩家数量训练优化的独立模型（使用 Stable-Baselines3 + PPO）：

```bash
# 自博弈训练（推荐，对手池 + 课程学习）
python -m rl_specialized.training.train_selfplay \
  --num_players 2 \
  --timesteps 2000000 \
  --snapshot_freq 200000

# 对抗规则智能体训练
python -m rl_specialized.training.train_specialized \
  --num_players 2 \
  --timesteps 200000

# 使用 TensorBoard 监控训练
tensorboard --logdir runs/rl_specialized
```

**特点**：
- 玩家数量特定的动作空间（如 2 人局仅 97 个动作）
- 合法动作掩码在 logits 层生效
- VecNormalize 观测/奖励归一化
- 学习率/裁剪范围线性衰减
- KL 散度约束（target_kl=0.03）
- 势能奖励整形（可选，默认开启）

**模型保存位置**：
- 自博弈：`runs/rl_selfplay/best_model/best_model.zip` 和 `snapshots/`
- 规则对抗：`runs/rl_specialized/best_model/best_model.zip`

**PPO 模型评估**：

```bash
# 自动评估最佳模型（默认包含5个对手：Random、Conservative、Aggressive、Heuristic、LLM）
python -m rl_specialized.evaluator --num-games 100

# 指定模型路径
python -m rl_specialized.evaluator \
  --model-path runs/rl_selfplay/best_model/best_model.zip \
  --num-games 100

# 排除 LLM 对手（仅测试前4个对手）
python -m rl_specialized.evaluator --num-games 100 --no-llm

# JSON 输出模式（适合批处理）
python -m rl_specialized.evaluator --num-games 100 --json

# 保存评估结果
python -m rl_specialized.evaluator --num-games 100 --save-results
```

**评估对手列表**（共5个）：
1. **RandomAgent** - 随机策略（性能下界）
2. **ConservativeAgent** - 保守策略（低挑战阈值）
3. **AggressiveAgent** - 激进策略（高挑战阈值）
4. **HeuristicRuleAgent** - 启发式规则策略
5. **OptimizedLLMAgent** - 优化 LLM 策略（默认启用，可通过 `--no-llm` 关闭）

---

## DQN技术特性

### 1. 组合动作 Q 值 + 合法动作掩码
- **问题**：传统多头输出（主动作 / 模式 / 数量 / 点数分开预测）容易产生非法组合（如首手开牌、飞模式叫 1）
- **解决方案**：
  - 对完整动作组合计算统一 Q 值：`Q_combined = Q_主动作 + Q_模式 + Q_数量 + Q_点数`
  - 构建合法动作掩码（2×总骰子数×6 的布尔张量），将非法动作 Q 值设为 `-inf`
  - 从源头过滤无效动作，减少探索浪费

### 2. Double DQN + n-step 回报
- **Double DQN**：在线网络选择动作，目标网络评估 Q 值，减少过高估计偏差
- **n-step 回报**（默认 n=3）：加速终局奖励回传到前期步骤，对短局稀疏奖励博弈尤为重要
- **公式**：R_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n Q_target(s_{t+n}, argmax Q_online)

### 3. 势能奖励整形（Potential-Based Reward Shaping）
- **动机**：替代硬性的"前 K 步禁止开牌"规则，降低超参敏感性
- **方法**：
  - 根据"上次叫点成立概率"构造势能函数 Φ(s)
  - 整形奖励：F = β × (γΦ(s') - Φ(s))，其中 β=0.2
  - 鼓励合理抬价，抑制虚高叫点或过早开牌
- **理论保证**：势能整形不改变最优策略（Ng et al., 1999）

### 4. 对手课程学习与类型选择
- **三种对手类型**（通过 `--opponent_type` 参数控制）：
  - `mixed`（默认）：动态混合启发式对手和自博弈对手
  - `heuristic`：纯启发式对手（挑战阈值随训练线性提升）
  - `selfplay`：纯自博弈对手（使用冻结的历史策略）
- **启发式对手课程**：挑战阈值随训练线性提升（从 0.3 → 0.7），模拟对手逐渐变强
- **自博弈对手池**：每隔一定 episode 冻结当前策略作为对手，构建多样化对手池
- **动态混合模式**：按比例采样启发式对手和冻结对手（默认各占 50%，通过 `--heuristic_ratio` 控制）

### 5. 扩展状态表示（22 维）+ 历史长度控制
为缓解部分可观测（POMDP）问题，状态向量包含：
- 手牌点数分布（6 维）
- 游戏基本信息（玩家数、当前玩家、首手玩家标记等，4 维）
- 上次叫点信息（模式、数量、点数，3 维）
- **新增特征**（9 维）：
  - 上次叫点成立概率（基于手牌和二项分布估计）
  - 最近 3 次数量变化
  - 模式切换历史
  - 当前是否为首手标记
- **历史长度控制**（通过 `--history_len` 参数）：
  - 默认 0：不裁剪，暴露完整游戏历史
  - 设置为正整数：仅暴露最近 N 条历史记录
  - 用途：研究记忆长度对 POMDP 决策的影响（消融实验）

### 6. Polyak 软目标更新
- 采用 τ=0.005 的软更新替代硬拷贝：θ_target ← τθ + (1-τ)θ_target
- 目标网络更平滑，减少训练震荡

---

## DQN算法说明

### 为何不使用 RNN？

代码中预留了 RNN 接口（`hidden_state`、`reset_hidden_state()` 等），但**默认未启用**：
- **原因**：前期实验发现 RNN 收敛不稳定，且计算开销大
- **现状**：通过扩展状态维度（22 维）+ n-step 回报缓解 POMDP 问题
- **未来**：保留接口便于后续升级为 GRU/LSTM

### 部分可观测性（POMDP）处理

骗子骰子是**部分可观测博弈**（看不到对手骰子和意图），当前缓解手段：
1. **状态扩展**：显式编码叫点可信度、模式切换历史
2. **n-step 回报**：加速稀疏奖励传播，减少值函数震荡
3. **对手课程**：平滑样本分布，避免策略突变导致的不稳定

**局限性**：系统仍假设"扩展状态 ≈ 马尔可夫"，在极端策略切换下可能震荡，这也是未来引入 RNN 的动力。

### 超参数选择建议

| 参数 | 默认值 | 说明 | 调优建议 |
|-----|--------|------|---------|
| `learning_rate` | 0.0001 | 学习率 | 过高易震荡，过低收敛慢 |
| `n_step` | 3 | n-step 步长 | 2-5 为宜，过大延迟奖励传播 |
| `shaping_beta` | 0.2 | 势能整形系数 | 0.1-0.3，过大可能掩盖真实奖励 |
| `reward_shaping` | True | 是否启用势能塑形 | 关闭后退化为纯稀疏奖励训练 |
| `ddqn` | True | 是否启用 Double DQN | 关闭后退化为 Vanilla DQN |
| `min_epsilon` | 0.1 | 探索率下限 | 0.05-0.15，过低易收敛到局部最优 |
| `target_soft_tau` | 0.005 | 目标网络更新率 | 0.001-0.01，过大目标网络变化快 |
| `opponent_type` | mixed | 对手类型 | heuristic 稳定但单一，selfplay 多样但初期弱，mixed 平衡 |
| `history_len` | 0 | 历史长度限制 | 0=完整历史，3-5 研究记忆影响 |
| `gamma` | 0.99 | 折扣因子 | 固定，短局博弈不建议调整 |

---

## PPO策略技术特性与算法说明
详细文档请参考：[rl_specialized/README.md](rl_specialized/README.md)

---

## Baseline实验支持

### LLM Agent（通义千问）

基于大语言模型的智能体，用于评估RL模型的人类水平对比基准。

**核心特性**：
- **API集成**：使用阿里云通义千问（Qwen）API，OpenAI兼容接口
- **API可用性检查**：自动检测API Key有效性，无API Key时自动使用fallback模式
- **超时与异常处理**：API调用设置10秒超时，异常时自动fallback，确保评估不中断
- **三层合法性验证**：
  - Layer 1: Prompt Engineering（系统提示包含完整规则+合法动作提示）
  - Layer 2: Rule-based Validation（使用`utils.is_strictly_greater()`验证）
  - Layer 3: Fallback Strategy（验证失败时从合法动作集中选择）
- **100%合法性保证**：通过三层验证确保输出动作始终合法
- **测试友好**：内置统计追踪（API调用次数、延迟、非法率、fallback次数等）

**快速开始**：

```bash
# 1. 设置API Key
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxx"

# 2. 可选：选择模型（默认qwen-max）
export DASHSCOPE_MODEL="qwen-max"  # 或 qwen-plus, qwen-turbo

# 3. 使用示例
python -c "
from agents.llm_agent import LLMAgent
agent = LLMAgent('llm_player', num_players=2, temperature=0.5, enable_stats=True, use_api=True)
# ... 在游戏中使用
stats = agent.get_stats()
print(f'非法动作率: {stats["illegal_rate"]:.2%}')
print(f'API可用性: {agent.has_api}')
print(f'使用API: {agent.use_api}')
"
```

**文档**：参见 `agents/LLM_AGENT_README.md`

### Baseline Agents

用于性能对比的基准智能体集合（位于 `agents/baseline_agents.py`）：

#### 1. RandomAgent
从所有合法动作中均匀随机选择，提供最弱性能下界。

**特性**：
- 100%合法性保证（使用`utils.get_legal_actions()`）
- 支持随机种子（可重复实验）
- 内置统计功能（动作分布、猜测率等）

**使用示例**：

```python
from agents.baseline_agents import RandomAgent

# 创建agent
agent = RandomAgent(agent_id="random_0", num_players=2, seed=42)

# 获取动作
action = agent.get_action(observation)

# 查看统计
stats = agent.get_stats()
print(f"猜测比例: {stats['guess_rate']:.2%}")
```

**应用场景**：
- 评估环境公平性（随机agent胜率应接近1/n）
- 提供性能下界（任何智能agent都应超越随机策略）
- 检测对手agent的可利用弱点

#### 2. ConservativeAgent
保守型策略智能体，倾向于挑战而非继续猜测。

**特性**：
- **保守挑战策略**：使用较低的信心阈值（默认0.5），更容易挑战对手
- **最小猜测策略**：必须猜测时选择最小合法猜测（count最小，face最小）
- **概率估计**：基于期望骰子数判断对手猜测是否可信
- **100%合法性保证**：使用`utils.get_legal_actions()`保证所有动作合法

**使用示例**：

```python
from agents.baseline_agents import ConservativeAgent

# 创建agent（可自定义信心阈值）
agent = ConservativeAgent(
    agent_id="conservative_0",
    num_players=2,
    confidence_threshold=0.5  # 越小越容易挑战
)

# 获取动作
action = agent.get_action(observation)

# 查看统计（包含挑战率）
stats = agent.get_stats()
print(f"挑战率: {stats['challenge_rate']:.2%}")
```

**应用场景**：
- 作为保守型baseline，评估激进策略的优势
- 测试环境对保守策略的奖励设计
- 提供不同风险偏好的性能对比

#### 3. AggressiveAgent
激进型策略智能体，倾向于猜测而非挑战。

**特性**：
- **激进猜测策略**：使用较高的信心阈值（默认2.0），很少挑战对手
- **大胆猜测策略**：基于手牌选择较大的合法猜测
- **手牌优化**：优先选择手里较多的点数进行猜测
- **100%合法性保证**：使用`utils.get_legal_actions()`保证所有动作合法

**使用示例**：

```python
from agents.baseline_agents import AggressiveAgent

# 创建agent（可自定义信心阈值）
agent = AggressiveAgent(
    agent_id="aggressive_0",
    num_players=2,
    confidence_threshold=2.0  # 越大越不容易挑战
)

# 获取动作
action = agent.get_action(observation)

# 查看统计
stats = agent.get_stats()
print(f"挑战率: {stats['challenge_rate']:.2%}")
```

**应用场景**：
- 作为激进型baseline，评估保守策略的必要性
- 测试环境对风险承担的奖励
- 提供不同风格的性能对比

#### 4. OptimizedLLMAgent
优化的LLM智能体，继承`LLMAgent`并使用更确定性的参数。

**特性**：
- **继承LLMAgent**：复用完整的三层验证机制
- **优化参数**：使用更低的temperature（0.3 vs 0.7），提高决策确定性
- **API控制**：新增`use_api`参数，可控制是否使用API（默认True）
- **100%合法性保证**：继承父类的三层验证（JSON解析 → 规则验证 → Fallback）
- **统计追踪**：默认启用统计功能

**使用示例**：

```python
from agents.baseline_agents import OptimizedLLMAgent

# 需要先设置环境变量
# export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxx"

# 创建agent（可自定义温度和API使用）
agent = OptimizedLLMAgent(
    agent_id="llm_0",
    num_players=2,
    temperature=0.3,  # 可自定义温度
    use_api=True      # 控制是否使用API（默认True）
)

# 获取动作（自动调用API）
action = agent.get_action(observation)

# 查看统计（包含API调用信息）
stats = agent.get_stats()
print(f"平均延迟: {stats['avg_latency']:.2f}s")
print(f"非法率: {stats['illegal_rate']:.2%}")
```

**应用场景**：
- 作为高级baseline，评估RL模型相对于human-like策略的优势
- 测试LLM在博弈论游戏中的表现
- 提供可解释的决策过程（reasoning字段）

### Baseline实验对比

运行多agent对局来评估性能差异：

```python
from agents.baseline_agents import RandomAgent, ConservativeAgent, AggressiveAgent
from env import LiarDiceEnv

# 创建环境和agents
env = LiarDiceEnv(num_players=2)
agents = {
    'player_0': ConservativeAgent('player_0', num_players=2),
    'player_1': AggressiveAgent('player_1', num_players=2)
}

# 运行多局游戏
for game in range(100):
    env.reset()
    for agent_id in env.agent_iter(100):
        observation = env.observe(agent_id)
        if env.terminations[agent_id] or env.truncations[agent_id]:
            action = None
        else:
            action = agents[agent_id].get_action(observation)
        env.step(action)
        if all(env.terminations.values()):
            break

# 查看统计
for agent_id, agent in agents.items():
    stats = agent.get_stats()
    print(f"{agent_id}: {stats}")
```

---

## 开发指南

### 代码规范

- **代码语言**：所有代码（变量名、函数名、类名）使用英文
- **注释语言**：所有代码注释使用中文
- **文档语言**：所有文档（.md文件、docstring）使用中文
- **命名规范**：
  - 类名：PascalCase（如`DQNAgent`）
  - 函数名/变量名：snake_case（如`get_legal_actions`）
  - 常量：UPPER_SNAKE_CASE（如`MAX_EPISODES`）

### 合法性一致性

**关键**：修改游戏规则或状态表示时，确保以下文件保持一致：
1. `env.py`：`LiarDiceEnv._is_strictly_greater()` - 权威规则实现
2. `utils.py`：`is_strictly_greater()` 和 `get_legal_actions()` - 必须匹配env.py逻辑
3. `agents/DQN_agent.py`：`_build_guess_legal_mask()` - 必须遵循相同规则
4. `rl_specialized/action_spaces/base.py`：如使用专用动作空间

### 状态维度变更

如果修改状态表示（当前DQN使用22维）：
- 更新`DQNAgent`输入维度
- 更新`env.py`中的observation构建
- 如使用`rl_specialized`，同步更新`StateEncoder`
- 作废旧模型检查点（重命名或删除）
- 更新README.md文档