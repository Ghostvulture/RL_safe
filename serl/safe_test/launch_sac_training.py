#!/usr/bin/env python3
"""
正确的GO2 SAC训练启动脚本 - 确保WandB可视化工作
"""

import subprocess
import time
import os
import sys
import signal

def run_sac_training():
    """启动完整的SAC训练"""
    
    print("🚀 启动GO2 SAC训练 (带WandB可视化)")
    print("=" * 50)
    
    exp_name = "go2_sac_working"
    max_steps = 5000  # 较短的测试
    
    processes = []
    
    try:
        # 启动Learner
        print("🧠 启动Learner...")
        learner_cmd = [
            "python", 
            "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py",
            "--learner",  # 关键参数
            "--env", "GO2-v0",
            "--exp_name", exp_name,
            "--max_steps", str(max_steps),
            "--batch_size", "64",  # 较小batch用于快速测试
            "--replay_buffer_capacity", "10000",
            "--training_starts", "500",
            "--log_period", "25",  # 更频繁的记录
            "--random_steps", "200"
        ]
        
        learner_env = os.environ.copy()
        learner_process = subprocess.Popen(
            learner_cmd,
            cwd="/home/xyz/Desktop/xluo/RL_safe/serl/safe_test",
            env=learner_env
        )
        processes.append(("Learner", learner_process))
        print("✓ Learner进程启动")
        
        # 等待learner初始化
        print("⏳ 等待Learner初始化...")
        time.sleep(8)
        
        # 启动Actor
        print("🎭 启动Actor...")
        actor_cmd = [
            "python",
            "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py", 
            "--actor",  # 关键参数
            "--env", "GO2-v0",
            "--exp_name", exp_name,
            "--max_steps", str(max_steps),
            "--max_traj_length", "200",  # 较短轨迹
            "--random_steps", "200",
            "--log_period", "25"
        ]
        
        actor_env = os.environ.copy()
        actor_process = subprocess.Popen(
            actor_cmd,
            cwd="/home/xyz/Desktop/xluo/RL_safe/serl/safe_test",
            env=actor_env
        )
        processes.append(("Actor", actor_process))
        print("✓ Actor进程启动")
        
        print(f"\n📈 训练开始! (总步数: {max_steps})")
        print("🔗 WandB可视化:")
        print("   项目: go2_sac_training")
        print("   查看: https://wandb.ai/")
        print("\n⏳ 训练运行中... (按Ctrl+C停止)")
        
        # 监控进程
        while True:
            # 检查进程状态
            running_processes = []
            for name, process in processes:
                if process.poll() is None:
                    running_processes.append((name, process))
                else:
                    print(f"ℹ️  {name}进程已结束 (返回码: {process.returncode})")
            
            processes = running_processes
            
            if not processes:
                print("✅ 所有进程已完成")
                break
                
            time.sleep(3)
        
    except KeyboardInterrupt:
        print("\n🛑 用户停止训练")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        
    finally:
        # 清理进程
        print("\n🧹 清理进程...")
        for name, process in processes:
            if process.poll() is None:
                print(f"⏹️  停止{name}...")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(f"🔪 强制结束{name}...")
                    process.kill()
        
        print("✅ 清理完成")
        print("\n🌐 查看训练结果:")
        print("   WandB: https://wandb.ai/")
        print("   项目: go2_sac_training")
        print(f"   运行: SAC_GO2_*_{exp_name}")

if __name__ == "__main__":
    run_sac_training()
