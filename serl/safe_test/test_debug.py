#!/usr/bin/env python3
"""
测试单独的learner和actor - 检查WandB输出
"""

import subprocess
import sys
import time
import os

def test_learner():
    """测试learner模式"""
    print("🧠 测试Learner模式...")
    
    cmd = [
        "python", 
        "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py",
        "--learner",  # 关键：指定learner模式
        "--env", "GO2-v0",
        "--exp_name", "test_learner",
        "--max_steps", "10",  # 很少的步数用于快速测试
        "--batch_size", "256",
        "--training_starts", "5",
        "--log_period", "2",
        "--debug"  # 先用debug模式避免WandB连接问题
    ]
    
    print("🔧 命令:", " ".join(cmd))
    
    try:
        # 设置环境和路径
        env = os.environ.copy()
        
        result = subprocess.run(
            cmd,
            cwd="/home/xyz/Desktop/xluo/RL_safe/serl/safe_test",
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        print("📤 标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  标准错误:")
            print(result.stderr)
            
        print(f"🏁 退出代码: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_actor():
    """测试actor模式"""
    print("\n🎭 测试Actor模式...")
    
    cmd = [
        "python",
        "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim/async_sac_state_sim.py", 
        "--actor",  # 关键：指定actor模式
        "--env", "GO2-v0",
        "--exp_name", "test_actor", 
        "--max_steps", "5",  # 很少的步数
        "--random_steps", "3",
        "--log_period", "2",
        "--debug"  # 先用debug模式
    ]
    
    print("🔧 命令:", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/home/xyz/Desktop/xluo/RL_safe/serl/safe_test",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("📤 标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  标准错误:")
            print(result.stderr)
            
        print(f"🏁 退出代码: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔍 GO2 SAC 脚本测试")
    print("=" * 50)
    
    # 测试learner
    learner_ok = test_learner()
    
    # 测试actor (单独运行，因为需要learner server)
    # actor_ok = test_actor()
    
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"Learner: {'✅ 通过' if learner_ok else '❌ 失败'}")
    # print(f"Actor: {'✅ 通过' if actor_ok else '❌ 失败'}")
    
    if learner_ok:
        print("\n💡 下一步:")
        print("1. Learner测试通过，可以尝试移除--debug参数启用WandB")
        print("2. 然后同时运行learner和actor进行完整训练")
    else:
        print("\n🔧 需要修复Learner问题才能继续")

if __name__ == "__main__":
    main()
