# 改进的GO2稀疏奖励系统

## 问题分析

你观察到机器人在简单奖励下往前倾倒的问题很常见。这是因为原始的稀疏奖励只关注前向速度，机器人发现"摔倒式前进"可以快速获得速度奖励，但没有学会正确的步行行为。

## 新的奖励设计

我已经为你添加了一个多组件的稀疏奖励系统，专门解决这个问题：

### 1. Hip运动奖励 (`_compute_hip_movement_reward`)
**目的：** 鼓励腿部运动，防止机器人只是倾倒前进

**机制：**
- **Hip速度奖励：** 鼓励hip关节有适度的运动（0.5-3.0 rad/s）
- **协调性奖励：** 鼓励正确的步态模式
  - 对角线协调：FL与RR同相，FR与RL同相
  - 左右交替：前腿和后腿交替运动

**为什么有效：** 
- 强制机器人使用腿部关节来移动
- 避免"僵硬倾倒"的策略

### 2. 稳定性奖励 (`_compute_stability_reward`)
**目的：** 防止倾倒，保持直立行走

**机制：**
- **高度稳定性：** 保持合理高度（0.25-0.45m）
- **姿态稳定性：** 限制roll和pitch角度
- **角速度稳定性：** 避免剧烈旋转

### 3. 接触因子 (`_compute_contact_factor`)
**目的：** 只有保持稳定时才能获得奖励

**机制：**
- 基本稳定性检查
- 根据稳定程度调整奖励倍数（0.1-1.0）

### 4. 动作平滑性奖励 (`_compute_action_smoothness_reward`)
**目的：** 鼓励平滑的关节运动

**机制：**
- 惩罚过于剧烈的动作变化
- 鼓励自然的运动模式

## 奖励组合策略

```python
base_reward = velocity_reward - yaw_rate_penalty
bonus_rewards = (hip_movement_reward * 0.5 + 
                stability_reward * 1.0 + 
                action_smoothness_reward * 0.3)

total_reward = (base_reward + bonus_rewards) * contact_factor * 10.0
```

**权重说明：**
- `hip_movement_reward * 0.5`: 中等权重，鼓励腿部运动
- `stability_reward * 1.0`: 高权重，强调稳定性
- `action_smoothness_reward * 0.3`: 低权重，辅助平滑运动
- `contact_factor`: 乘数因子，不稳定时大幅降低奖励

## 使用方法

确保你的配置文件中启用了稀疏奖励：

```python
# 在simple_go2_config.py中
class env:
    use_sparse_reward = True
```

## 调试输出

每20步会打印详细的奖励分解：

```
SPARSE REWARD: vel=0.234, adj_vel=0.228, vel_rew=0.456
  hip_mov=0.123, stability=0.789, contact=0.8
  yaw_pen=0.012, smooth=0.345, total=4.44
```

## 参数调整建议

如果机器人仍然倾倒，可以：

1. **增加稳定性权重：** `stability_reward * 1.5`
2. **增加hip运动权重：** `hip_movement_reward * 0.8`
3. **降低速度阈值：** 将目标速度从0.5降到0.3
4. **收紧稳定性限制：** 减小允许的角度范围

如果机器人太保守不敢动，可以：

1. **增加速度奖励权重**
2. **放宽稳定性限制**
3. **增加hip运动的期望速度范围**

## 核心创新

这个系统的关键在于：
1. **强制使用腿部：** hip运动奖励确保机器人必须动腿
2. **平衡约束：** 稳定性奖励防止极端策略
3. **渐进学习：** 接触因子确保只有稳定时才能获得大奖励

这样机器人就必须学会在保持稳定的同时使用腿部运动来前进，而不是简单地倾倒。
