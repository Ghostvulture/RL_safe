#!/usr/bin/env python3
"""
单独测试Actor的WandB输出 - 不依赖Learner
"""

import sys
import os
import time
import numpy as np
import wandb

# 添加路径
sys.path.append('/home/xyz/Desktop/xluo/RL_safe/serl/safe_test/go2')

def test_actor_wandb():
    """测试Actor单独运行时的WandB输出"""
    
    print("🎭 单独测试Actor WandB输出...")
    
    try:
        # 导入GO2环境
        from go2_gym_env import make_go2_env
        print("✓ GO2环境导入成功")
        
        # 初始化WandB - 使用新的项目名
        wandb.init(
            project="go2_actor_solo_test",  # 新的项目名
            name="actor_standalone_test",
            group="GO2_Actor_Testing",
            tags=["SAC", "GO2", "actor", "standalone"],
            config={
                "env": "GO2-v0",
                "algorithm": "SAC_Actor_Only",
                "test_episodes": 3,
                "max_episode_length": 100,
                "mode": "actor_standalone"
            }
        )
        print("✓ WandB初始化成功")
        print(f"🌐 项目: go2_actor_solo_test")
        print(f"🏃 运行: actor_standalone_test")
        
        # 创建环境
        env = make_go2_env(render_mode='rgb_array')
        print("✓ GO2环境创建成功")
        
        # 模拟Actor行为
        episode_count = 0
        total_steps = 0
        
        for episode in range(3):  # 3个episode
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            action_magnitudes = []
            
            print(f"📍 Episode {episode + 1}/3 开始...")
            
            for step in range(100):  # 每个episode最多100步
                # 随机动作 (模拟探索阶段)
                action = env.action_space.sample()
                action_magnitudes.append(np.linalg.norm(action))
                
                # 环境步进
                next_obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # 每10步记录一次
                if step % 10 == 0:
                    wandb.log({
                        "actor/step_reward": float(reward),
                        "actor/total_steps": total_steps,
                        "actor/episode": episode + 1,
                        "actor/episode_step": step,
                        "actor/action_magnitude": float(np.linalg.norm(action)),
                        "actor/exploration_mode": True,
                    }, step=total_steps)
                
                if done or truncated:
                    print(f"  Episode结束: 步数={episode_length}, 奖励={episode_reward:.3f}")
                    break
                
                obs = next_obs
            
            # Episode结束时的统计
            episode_count += 1
            avg_action_mag = np.mean(action_magnitudes) if action_magnitudes else 0.0
            
            wandb.log({
                "actor/episode_reward": float(episode_reward),
                "actor/episode_length": episode_length,
                "actor/episode_count": episode_count,
                "actor/avg_action_magnitude": avg_action_mag,
                "actor/episode_success": not done,  # 没有done说明成功完成
            }, step=total_steps)
            
            print(f"✓ Episode {episode + 1} 完成: 奖励={episode_reward:.3f}, 步数={episode_length}")
            time.sleep(1)  # 稍微暂停
        
        print(f"\n📊 测试完成!")
        print(f"总Episodes: {episode_count}")
        print(f"总Steps: {total_steps}")
        
        # 结束WandB
        wandb.finish()
        env.close()
        
        print(f"\n🌐 查看结果:")
        print(f"项目: go2_actor_solo_test")
        print(f"运行: actor_standalone_test")
        print(f"链接: https://wandb.ai/")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 GO2 Actor 单独WandB测试")
    print("=" * 50)
    
    success = test_actor_wandb()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Actor WandB测试成功!")
        print("现在你可以在网页上看到actor的独立运行数据了")
    else:
        print("❌ 测试失败，请检查错误信息")

if __name__ == "__main__":
    main()
