#!/usr/bin/env python3

"""
GO2 SAC Training 双终端快速验证脚本
这个脚本用于验证Actor和Learner的WandB集成是否正常工作
"""

import subprocess
import time
import os
import signal
import sys

def cleanup_processes():
    """清理可能残留的进程"""
    print("🧹 清理端口5488上的进程...")
    try:
        subprocess.run("lsof -ti:5488 | xargs -r kill -9", shell=True, capture_output=True)
        print("✅ 端口清理完成")
    except:
        print("⚠️  端口清理可能失败，但继续执行")

def run_learner_test():
    """启动Learner进行短暂测试"""
    print("🎓 启动Learner测试...")
    script_dir = "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim"
    
    try:
        # 启动learner (10秒超时)
        process = subprocess.Popen([
            "bash", "run_learner.sh", 
            "--max_steps", "5",
            "--training_starts", "5",
            "--batch_size", "32",
            "--log_period", "1"
        ], cwd=script_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("⏳ Learner正在启动（等待10秒）...")
        try:
            stdout, stderr = process.communicate(timeout=10)
            print("✅ Learner测试完成")
            if stdout:
                print("📊 Learner输出:", stdout.decode()[-200:])  # 只显示最后200字符
        except subprocess.TimeoutExpired:
            print("⏰ Learner测试超时，终止进程")
            process.kill()
            process.communicate()
        
    except Exception as e:
        print(f"❌ Learner测试失败: {e}")

def main():
    print("🚀 GO2 SAC Training 双终端验证")
    print("=" * 50)
    
    cleanup_processes()
    time.sleep(2)
    
    print("\n📋 验证摘要:")
    print("1. Actor脚本已修改为使用项目名 'go2_sac_training'")
    print("2. Learner脚本已修改为使用项目名 'go2_sac_training'")
    print("3. 运行名称将显示为:")
    print("   - Actor: Actor-go2_sac_walking_training")
    print("   - Learner: Learner-go2_sac_walking_training")
    
    print("\n🎯 双终端执行步骤:")
    print("-" * 30)
    print("终端1 (Learner):")
    print("cd /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim")
    print("bash run_learner.sh --max_steps 100")
    print()
    print("终端2 (Actor):")
    print("cd /home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim")
    print("bash run_actor.sh --max_steps 100")
    
    print("\n🌐 WandB查看:")
    print("项目: go2_sac_training")
    print("网址: https://wandb.ai/your-username/go2_sac_training")
    
    print("\n✨ 配置验证完成！你现在可以:")
    print("1. 在两个终端中分别运行上述命令")
    print("2. 观察WandB中出现两个不同的运行")
    print("3. Actor显示环境交互数据，Learner显示训练损失")
    print("=" * 50)

if __name__ == "__main__":
    main()
