#!/usr/bin/env python3

"""
WandB监控变量快速查看脚本
用于展示GO2 SAC训练中所有监控的变量
"""

def display_wandb_metrics():
    print("🎯 GO2 SAC Training WandB监控变量总览")
    print("=" * 60)
    
    actor_metrics = {
        "Episode级别指标 (每episode结束记录)": [
            "actor/episode_reward - 单个episode总奖励",
            "actor/episode_length - episode步数长度", 
            "actor/episode_count - 累计episode数量",
            "actor/total_steps - 累计环境交互步数",
            "actor/avg_action_magnitude - 该episode平均动作幅度",
            "actor/exploration_phase - 是否随机探索阶段"
        ],
        "定期统计指标 (每log_period步记录)": [
            "actor/avg_episode_reward_10 - 最近10个episode平均奖励",
            "actor/max_episode_reward - 历史最高episode奖励",
            "actor/min_episode_reward - 历史最低episode奖励", 
            "actor/avg_episode_length_10 - 最近10个episode平均长度",
            "actor/steps_per_second - 环境交互速度",
            "actor/replay_buffer_size - 经验回放缓冲区大小"
        ],
        "评估指标 (每eval_period步记录)": [
            "eval/average_return - 评估episode平均奖励",
            "eval/success_rate - 评估成功率",
            "eval/average_length - 评估episode平均长度"
        ]
    }
    
    learner_metrics = {
        "SAC算法核心指标": [
            "sac/actor_loss - Actor网络损失",
            "sac/critic_loss - Critic网络损失",
            "sac/temperature_loss - 温度参数损失",
            "sac/temperature - 当前温度参数值",
            "sac/entropy - 策略熵值"
        ],
        "训练进度指标": [
            "learner/update_steps - 网络更新步数",
            "learner/replay_buffer_size - 经验回放缓冲区大小"
        ],
        "性能计时指标": [
            "timer/sample_replay_buffer - 采样缓冲区耗时",
            "timer/train - 网络训练耗时",
            "timer/total - 总循环耗时"
        ]
    }
    
    print("\n🎮 ACTOR端监控指标")
    print("-" * 40)
    for category, metrics in actor_metrics.items():
        print(f"\n📊 {category}:")
        for metric in metrics:
            print(f"  • {metric}")
    
    print("\n🧠 LEARNER端监控指标")
    print("-" * 40)
    for category, metrics in learner_metrics.items():
        print(f"\n📊 {category}:")
        for metric in metrics:
            print(f"  • {metric}")
    
    print("\n🔍 关键监控重点")
    print("-" * 40)
    print("✅ 训练成功指标:")
    print("  • actor/episode_reward ↗️ 应该逐渐增加")
    print("  • actor/episode_length → 应接近max_traj_length(500)")
    print("  • sac/actor_loss & sac/critic_loss → 应该逐渐收敛")
    print("  • actor/avg_episode_reward_10 → 显示平滑趋势")
    
    print("\n⚠️  异常监控指标:")
    print("  • sac/temperature → 过高/过低都有问题")
    print("  • actor/avg_action_magnitude → 异常高值表示策略不稳定")
    print("  • timer/* → 异常高值表示性能瓶颈")
    
    print("\n🚀 快速启动命令")
    print("-" * 40)
    print("终端1 (Learner):")
    print("cd /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim")
    print("bash run_learner.sh --max_steps 10000")
    print()
    print("终端2 (Actor):")
    print("cd /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim")
    print("bash run_actor.sh --max_steps 10000")
    print()
    print("🌐 WandB查看: https://wandb.ai/your-username/go2_sac_training")
    print("=" * 60)

if __name__ == "__main__":
    display_wandb_metrics()
