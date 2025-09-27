#!/usr/bin/env python3
"""
最小化的WandB测试脚本 - 检查网页可视化功能
"""

import time
import numpy as np
import wandb

def main():
    """最小化的WandB测试"""
    print("🚀 启动WandB最小化测试...")
    
    # 初始化WandB
    wandb.init(
        project="go2_sac_minimal_test",
        name="minimal_test_run", 
        config={
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 100,
            "algorithm": "SAC",
            "env": "GO2-v0",
        },
        tags=["test", "minimal", "GO2", "SAC"]
    )
    
    print("✓ WandB初始化成功")
    print(f"📊 项目: go2_sac_minimal_test")
    print(f"🏃 运行名称: minimal_test_run")
    print(f"🌐 查看链接: https://wandb.ai/")
    
    # 模拟训练过程
    print("\n📈 开始模拟训练数据...")
    
    for step in range(100):
        # 模拟奖励数据 (随着训练逐渐提高)
        episode_reward = -50 + step * 0.8 + np.random.normal(0, 5)
        
        # 模拟SAC损失 (随着训练逐渐下降)
        actor_loss = 1.0 * np.exp(-step/30) + np.random.normal(0, 0.1)
        critic_loss = 2.0 * np.exp(-step/25) + np.random.normal(0, 0.2)
        
        # 模拟其他指标
        episode_length = 500 + np.random.randint(-100, 100)
        exploration_noise = max(0.1, 1.0 - step/100)
        
        # 记录到WandB
        wandb.log({
            # 奖励相关
            "actor/episode_reward": episode_reward,
            "actor/episode_length": episode_length,
            "actor/avg_reward_10": np.mean([episode_reward + np.random.normal(0, 2) for _ in range(10)]),
            
            # SAC算法指标
            "sac/actor_loss": actor_loss,
            "sac/critic_loss": critic_loss,
            "sac/temperature": 0.2 + np.random.normal(0, 0.02),
            "sac/entropy": -1.5 + np.random.normal(0, 0.1),
            
            # 训练进度
            "training/step": step,
            "training/exploration_noise": exploration_noise,
            "training/replay_buffer_size": min(10000, step * 100),
            
        }, step=step)
        
        # 每10步打印一次进度
        if step % 10 == 0:
            print(f"步骤 {step:3d}/100 | 奖励: {episode_reward:6.2f} | Actor损失: {actor_loss:.4f}")
        
        # 稍微延迟以模拟真实训练
        time.sleep(0.1)
    
    print("\n✅ 训练模拟完成!")
    print("🔗 请访问 https://wandb.ai/ 查看可视化结果")
    print("📂 项目名称: go2_sac_minimal_test")
    print("📊 运行名称: minimal_test_run")
    
    # 完成WandB运行
    wandb.finish()
    
    print("\n💡 使用说明:")
    print("1. 在浏览器中访问 https://wandb.ai/")  
    print("2. 登录你的WandB账户")
    print("3. 找到项目 'go2_sac_minimal_test'")
    print("4. 点击运行 'minimal_test_run'")
    print("5. 查看图表和指标")

if __name__ == "__main__":
    main()
