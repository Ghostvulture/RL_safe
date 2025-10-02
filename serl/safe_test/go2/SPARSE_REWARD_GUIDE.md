# GO2 稀疏奖励使用指南

## 概述

我已经为你添加了一个稀疏奖励系统，类似于SAC fine-tuning代码中的简单设计。这个系统提供了两种奖励模式：

1. **复杂奖励模式**（默认）：使用多个奖励组件的原始系统
2. **稀疏奖励模式**：基于前向速度的简单奖励系统

## 稀疏奖励设计

稀疏奖励模仿了你提供的SAC fine-tuning代码：

```python
# 原始代码逻辑：
reward = rewards.tolerance(lin_vel * np.cos(pitch),
                           bounds=(target_vel, 2 * target_vel),
                           margin=2 * target_vel,
                           value_at_margin=0,
                           sigmoid='linear')
reward -= 0.1 * np.abs(drpy[-1])  # 偏航率惩罚
reward *= max(self._foot_contacts)  # 接触因子
reward *= 10.0  # 缩放因子
```

我的实现包括：
- **前向速度奖励**：使用tolerance函数，目标速度0.5 m/s，最佳范围0.5-1.0 m/s
- **偏航率惩罚**：-0.1 * |角速度z|
- **接触因子**：基于高度和姿态的简化接触检测
- **缩放因子**：最终奖励 × 10.0

## 使用方法

### 方法1：修改现有配置
在 `simple_go2_config.py` 中设置：
```python
class env:
    use_sparse_reward = True  # 启用稀疏奖励
```

### 方法2：使用新的稀疏配置文件
我已经创建了 `sparse_go2_config.py`，你可以这样使用：

```python
# 在你的训练脚本中
from sparse_go2_config import SparseGO2Config as GO2RoughCfg
```

### 方法3：动态切换
你也可以在运行时切换：
```python
# 切换到稀疏奖励
GO2RoughCfg.env.use_sparse_reward = True

# 切换回复杂奖励
GO2RoughCfg.env.use_sparse_reward = False
```

## 调试输出

稀疏奖励模式会每20步输出一次调试信息：
```
SPARSE REWARD: vel=0.234, adj_vel=0.228, vel_rew=0.456, yaw_pen=0.012, contact=1.0, total=4.44
```

## 参数调整

你可以在 `_compute_sparse_reward()` 函数中调整以下参数：
- `target_vel = 0.5`：目标前向速度
- `bounds=(target_vel, 2 * target_vel)`：最佳速度范围
- `margin=2 * target_vel`：容忍边界
- `yaw_rate_penalty = 0.1`：偏航率惩罚系数
- 接触检测阈值：高度 > 0.15m，角度 < 0.5 rad

## 优势

稀疏奖励的优势：
1. **简单直接**：只关注核心行为（前进）
2. **稳定训练**：减少奖励成分之间的冲突
3. **易于调试**：单一主要奖励信号
4. **快速收敛**：专注于主要任务目标

## 切换示例

```python
# 训练开始时使用复杂奖励进行基础学习
GO2RoughCfg.env.use_sparse_reward = False

# 训练后期切换到稀疏奖励进行fine-tuning
GO2RoughCfg.env.use_sparse_reward = True
```

现在你可以尝试这两种奖励模式，看看哪种对你的训练任务效果更好！
