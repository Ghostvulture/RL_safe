#!/usr/bin/env python3
"""
GO2 SAC训练启动脚本 - 包含完整WandB可视化
"""

import subprocess
import time
import os
import signal
import sys

def run_go2_sac_training():
    """启动GO2 SAC训练的learner和actor"""
    
    print("🚀 启动GO2 SAC训练...")
    print("📊 WandB项目: safe_go2")
    print("🌐 查看地址: https://wandb.ai/")
    
    # 训练参数
    exp_name = "go2_walking_v1" 
    max_steps = 10000  # 较短的测试运行
    
    processes = []
    
    try:
        # 启动Learner进程
        print("\n🧠 启动Learner进程...")
        learner_cmd = [
            "python", "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py",
            "--learner",
            "--env", "GO2-v0", 
            "--exp_name", exp_name,
            "--max_steps", str(max_steps),
            "--batch_size", "256",
            "--replay_buffer_capacity", "50000",  # 较小的缓冲区用于测试
            "--training_starts", "1000",
            "--log_period", "50",
            "--eval_period", "2000",
            "--random_steps", "500"
        ]
        
        learner_process = subprocess.Popen(
            learner_cmd,
            cwd="/home/xyz/Desktop/xluo/RL_safe/serl/safe_test"
        )
        processes.append(("Learner", learner_process))
        print("✓ Learner进程已启动")
        
        # 等待learner启动
        time.sleep(5)
        
        # 启动Actor进程  
        print("\n🎭 启动Actor进程...")
        actor_cmd = [
            "python", "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py",
            "--actor",
            "--env", "GO2-v0",
            "--exp_name", exp_name, 
            "--max_steps", str(max_steps),
            "--max_traj_length", "1000",
            "--random_steps", "500",
            "--log_period", "50"
        ]
        
        actor_process = subprocess.Popen(
            actor_cmd,
            cwd="/home/xyz/Desktop/xluo/RL_safe/serl/safe_test"
        )
        processes.append(("Actor", actor_process))
        print("✓ Actor进程已启动")
        
        print(f"\n📈 训练开始! 总步数: {max_steps}")
        print("🔗 WandB可视化:")
        print("   - 项目: safe_go2")
        print("   - 运行: SAC_GO2_learner_go2_walking_v1 & SAC_GO2_actor_go2_walking_v1")
        print("   - 地址: https://wandb.ai/")
        
        # 等待训练完成
        print("\n⏳ 等待训练完成... (按Ctrl+C停止)")
        
        while True:
            # 检查进程状态
            for name, process in processes:
                if process.poll() is not None:
                    print(f"✓ {name}进程已完成")
                    
            # 如果所有进程都完成了
            if all(process.poll() is not None for _, process in processes):
                print("\n🎉 训练完成!")
                break
                
            time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断训练...")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        
    finally:
        # 清理所有进程
        print("\n🧹 清理进程...")
        for name, process in processes:
            if process.poll() is None:
                print(f"⏭️  终止{name}进程...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"🔪 强制结束{name}进程...")
                    process.kill()
        
        print("✅ 清理完成")

if __name__ == "__main__":
    run_go2_sac_training()
