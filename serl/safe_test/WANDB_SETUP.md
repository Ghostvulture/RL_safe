# GO2 SAC Training WandB监控指南

本文档详细介绍了GO2四足机器人SAC训练过程中WandB监控的所有变量和指标。

## 🎯 项目结构

### WandB项目配置
- **项目名称**: `go2_sac_training`
- **Actor运行名**: `Actor-go2_sac_walking_training`
- **Learner运行名**: `Learner-go2_sac_walking_training`
- **分组**: `GO2_SAC_Training`

## 📊 监控变量详解

### 🎮 Actor端监控指标

Actor负责与环境交互，收集经验数据。以下是Actor端WandB记录的关键指标：

#### 🏃 Episode级别指标 (每个episode结束时记录)
| 变量名 | 描述 | 意义 |
|--------|------|------|

| `actor/episode_reward` | 单个episode的总奖励 | 衡量机器人在一个episode中的整体表现 |
| `actor/episode_length` | episode的步数长度 | 机器人能够持续行走的时间 |
| `actor/episode_count` | 累计完成的episode数量 | 训练进度指示器 |
| `actor/total_steps` | 累计环境交互步数 | 总的训练步数 |
| `actor/avg_action_magnitude` | 该episode中动作的平均幅度 | 动作激烈程度，用于监控动作平滑性 |
| `actor/exploration_phase` | 是否处于随机探索阶段 | `True`表示随机动作，`False`表示策略动作 |

#### 📈 定期统计指标 (每`log_period`步记录)
| 变量名 | 描述 | 意义 |
|--------|------|------|
| `actor/avg_episode_reward_10` | 最近10个episode的平均奖励 | 训练效果的平滑趋势 |
| `actor/max_episode_reward` | 历史最高episode奖励 | 最佳表现记录 |
| `actor/min_episode_reward` | 历史最低episode奖励 | 最差表现记录 |
| `actor/avg_episode_length_10` | 最近10个episode的平均长度 | 稳定性趋势 |
<!-- | `actor/steps_per_second` | 环境交互速度 (步/秒) | 训练效率指标 |
| `actor/replay_buffer_size` | 经验回放缓冲区大小 | 数据收集进度 | -->

#### 🎯 评估指标 (每`eval_period`步记录)
通过`evaluate`函数记录的评估结果，包含：
- 评估episode的平均奖励
- 评估episode的成功率
- 评估episode的平均长度

### 🧠 Learner端监控指标

Learner负责神经网络训练，优化策略和价值函数。以下是Learner端记录的指标：

#### 🔥 SAC算法核心指标
| 变量名 | 描述 | 意义 |
|--------|------|------|
| `sac/actor_loss` | Actor网络损失 | 策略网络优化程度 |
| `sac/critic_loss` | Critic网络损失 | 价值函数拟合误差 |
<!-- | `sac/temperature_loss` | 温度参数损失 | 熵正则化调节 |
| `sac/temperature` | 当前温度参数值 | 探索与利用平衡 | -->
| `sac/entropy` | 策略熵值 | 动作分布的随机性 |

#### ⚡ 训练进度指标
| 变量名 | 描述 | 意义 |
|--------|------|------|
| `learner/update_steps` | 网络更新步数 | 训练迭代次数 |
| `learner/replay_buffer_size` | 经验回放缓冲区当前大小 | 可用训练数据量 |

#### ⏱️ 性能计时指标
| 变量名 | 描述 | 意义 |
|--------|------|------|
| `timer/sample_replay_buffer` | 采样缓冲区耗时 | 数据加载效率 |
| `timer/train` | 网络训练耗时 | 训练计算效率 |
| `timer/total` | 总循环耗时 | 整体训练效率 |

## 🔍 监控重点指标

### 🎯 训练成功指标
1. **`actor/episode_reward`**: 应该随着训练逐渐增加
2. **`actor/episode_length`**: 稳定行走时应接近`max_traj_length`
3. **`sac/actor_loss`** 和 **`sac/critic_loss`**: 应该逐渐收敛
4. **`actor/avg_episode_reward_10`**: 显示训练的平滑趋势

### ⚠️ 异常监控指标
1. **`sac/temperature`**: 过高可能导致过度探索，过低可能导致收敛到局部最优
2. **`actor/avg_action_magnitude`**: 异常高值可能表示不稳定的策略
3. **`timer/*`**: 异常高的计时值可能表示性能瓶颈

## 🚀 使用指南

### 启动训练
```bash
# 终端1: 启动Learner
cd /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim
bash run_learner.sh --max_steps 10000

# 终端2: 启动Actor  
cd /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim
bash run_actor.sh --max_steps 10000
```

### WandB查看
- **网址**: https://wandb.ai/your-username/go2_sac_training
- **Actor面板**: 关注环境交互和episode统计
- **Learner面板**: 关注网络训练损失和优化指标

## 📋 关键参数配置

### Actor关键参数
- `random_steps`: 1000 (随机探索步数)
- `max_traj_length`: 500 (最大episode长度)
- `steps_per_update`: 50 (多少步更新一次网络)
- `log_period`: 50 (日志记录间隔)

### Learner关键参数
- `training_starts`: 1000 (开始训练的最少数据量)
- `batch_size`: 128 (训练批次大小)
- `critic_actor_ratio`: 4 (critic与actor更新比例)

## 🎨 WandB可视化建议

### 推荐图表配置
1. **训练曲线**: `actor/episode_reward` vs 训练步数
2. **稳定性**: `actor/episode_length` vs 训练步数  
3. **学习进度**: `sac/actor_loss`, `sac/critic_loss` vs 更新步数
4. **探索情况**: `sac/entropy`, `sac/temperature` vs 更新步数
5. **效率监控**: `actor/steps_per_second`, `timer/total` vs 时间

### 自定义面板
建议创建两个自定义面板：
- **Environment Panel**: 包含所有`actor/*`指标
- **Training Panel**: 包含所有`sac/*`和`learner/*`指标

## 🔧 故障排除

### 常见问题
1. **WandB显示离线**: 检查网络连接和API密钥
2. **指标不更新**: 确认没有使用`--debug`标志
3. **Actor连接失败**: 确保Learner先启动并监听端口5488

### 调试模式
使用`--debug`标志可以禁用WandB日志，用于本地调试：
```bash
bash run_actor.sh --max_steps 100 --debug
```

## 📊 实际监控示例

### 成功训练的典型指标趋势

#### Actor端期望趋势
- `actor/episode_reward`: 从负值逐渐上升到正值
- `actor/episode_length`: 从较短逐渐增加到接近500步
- `actor/avg_action_magnitude`: 保持在合理范围(0.5-2.0)

#### Learner端期望趋势  
- `sac/actor_loss`: 初期较高，逐渐收敛
- `sac/critic_loss`: 随着网络学习逐渐降低
- `sac/temperature`: 自适应调节，通常在0.1-1.0范围

### 异常情况识别
- **奖励不增长**: 可能是学习率过高或环境奖励设计问题
- **episode长度太短**: 机器人频繁摔倒，需要调整奖励函数
- **损失震荡**: 可能需要降低学习率或增加批次大小

## 🎯 监控最佳实践

### 日常监控重点
1. **每日检查**: `actor/episode_reward`趋势
2. **每周分析**: 损失函数收敛情况
3. **性能优化**: 关注`timer/*`指标识别瓶颈

### 实验对比
利用WandB的实验对比功能：
- 对比不同超参数设置的效果
- 分析不同奖励函数设计的影响
- 评估网络架构改进的效果

---

📝 **更新日期**: 2025-09-27  
🏷️ **版本**: v2.0  
👨‍💻 **维护者**: GO2 SAC Training Team

## 📞 技术支持

如有问题请参考：
1. [WandB官方文档](https://docs.wandb.ai/)
2. [SAC算法论文](https://arxiv.org/abs/1801.01290)
3. 项目Issue页面

## 📋 关键参数配置

### Actor关键参数
- `random_steps`: 1000 (随机探索步数)
- `max_traj_length`: 500 (最大episode长度)
- `steps_per_update`: 50 (多少步更新一次网络)
- `log_period`: 50 (日志记录间隔)

### Learner关键参数
- `training_starts`: 1000 (开始训练的最少数据量)
- `batch_size`: 128 (训练批次大小)
- `critic_actor_ratio`: 4 (critic与actor更新比例)

## 🎨 WandB可视化建议

### 推荐图表配置
1. **训练曲线**: `actor/episode_reward` vs 训练步数
2. **稳定性**: `actor/episode_length` vs 训练步数  
3. **学习进度**: `sac/actor_loss`, `sac/critic_loss` vs 更新步数
4. **探索情况**: `sac/entropy`, `sac/temperature` vs 更新步数
5. **效率监控**: `actor/steps_per_second`, `timer/total` vs 时间

### 自定义面板
建议创建两个自定义面板：
- **Environment Panel**: 包含所有`actor/*`指标
- **Training Panel**: 包含所有`sac/*`和`learner/*`指标

## 🔧 故障排除

### 常见问题
1. **WandB显示离线**: 检查网络连接和API密钥
2. **指标不更新**: 确认没有使用`--debug`标志
3. **Actor连接失败**: 确保Learner先启动并监听端口5488

### 调试模式
使用`--debug`标志可以禁用WandB日志，用于本地调试：
```bash
bash run_actor.sh --max_steps 100 --debug
    "algorithm": "SAC", 
    "batch_size": 256,
    "replay_buffer_capacity": 100000,
    "max_steps": 100000,
    "utd_ratio": 1,
    "seed": 42,
    "max_traj_length": 1000,
    "random_steps": 5000,
}
```

## 文件修改说明

### 主要修改的文件：
1. **async_sac_state_sim.py**: 
   - 添加了WandB导入和配置
   - 在actor循环中添加轮次级别的指标记录
   - 在learner循环中添加训练指标记录
   - 增强了项目配置和标签

### 2. 新增的辅助文件：
- **train_go2_sac.py**: 启动脚本，自动运行learner和actor进程
- **test_wandb_setup.py**: 测试脚本，验证WandB和环境配置

## 使用方法

### 1. 启动完整训练：
```bash
cd /home/xyz/Desktop/xluo/RL_safe/serl/safe_test
python train_go2_sac.py
```

### 2. 手动启动（分别运行）：

启动Learner：
```bash
cd /home/xyz/Desktop/xluo/RL_safe/serl/safe_test
python /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py \
    --learner \
    --env GO2-v0 \
    --exp_name go2_sac_test \
    --batch_size 256 \
    --max_steps 100000
```

启动Actor：
```bash  
cd /home/xyz/Desktop/xluo/RL_safe/serl/safe_test
python /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py \
    --actor \
    --env GO2-v0 \
    --exp_name go2_sac_test \
    --max_steps 100000 \
    --random_steps 5000
```

### 3. 测试WandB配置：
```bash
cd /home/xyz/Desktop/xluo/RL_safe/serl/safe_test  
python test_wandb_setup.py
```

## WandB仪表板查看

训练开始后，可以在以下位置查看实时可视化：
- 网址: https://wandb.ai/
- 项目: `go2_sac_training`
- 组: `GO2_SAC_Training`

## 关键可视化图表

建议关注的图表：
1. **奖励趋势**: `actor/episode_reward`, `actor/avg_episode_reward_10`
2. **SAC损失**: `sac/actor_loss`, `sac/critic_loss`  
3. **策略熵**: `sac/entropy`
4. **温度参数**: `sac/temperature`
5. **训练效率**: `actor/steps_per_second`
6. **轮次长度**: `actor/episode_length`

## 注意事项

1. **环境依赖**: 需要安装wandb包：`pip install wandb`
2. **账户配置**: 首次使用需要登录WandB账户：`wandb login`
3. **调试模式**: 使用`--debug`标志可以禁用WandB记录
4. **网络连接**: WandB需要网络连接来同步数据

## 故障排除

1. **WandB导入错误**: 确保安装了wandb包
2. **GO2环境错误**: 确保MuJoCo和GO2模型文件正确配置
3. **网络超时**: 检查网络连接或使用离线模式
