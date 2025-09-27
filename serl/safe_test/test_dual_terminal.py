#!/usr/bin/env python3

"""
测试双终端执行的示例脚本
这个脚本演示如何在两个终端中分别运行actor和learner
"""

import os
import subprocess
import time

print("🚀 GO2 SAC 双终端训练测试")
print("=" * 50)

# 脚本路径
script_dir = "/home/xyz/Desktop/xluo/RL_safe/serl/examples/async_sac_state_sim"
actor_script = os.path.join(script_dir, "run_actor.sh")
learner_script = os.path.join(script_dir, "run_learner.sh")

print(f"📍 Actor脚本路径: {actor_script}")
print(f"📍 Learner脚本路径: {learner_script}")

# 检查脚本是否存在
if not os.path.exists(actor_script):
    print(f"❌ Actor脚本不存在: {actor_script}")
    exit(1)

if not os.path.exists(learner_script):
    print(f"❌ Learner脚本不存在: {learner_script}")
    exit(1)

print("\n✅ 脚本文件检查通过")

# 显示脚本内容
print("\n📄 Actor脚本内容:")
print("-" * 30)
with open(actor_script, 'r') as f:
    print(f.read())

print("\n📄 Learner脚本内容:")
print("-" * 30)
with open(learner_script, 'r') as f:
    print(f.read())

print("\n🎯 使用指南:")
print("=" * 50)
print("1. 打开终端1，执行以下命令启动Learner:")
print(f"   cd {script_dir}")
print("   bash run_learner.sh --max_steps 100")
print()
print("2. 打开终端2，执行以下命令启动Actor:")
print(f"   cd {script_dir}")
print("   bash run_actor.sh --max_steps 100")
print()
print("3. 在WandB网页上查看结果:")
print("   项目名: go2_sac_training")
print("   Actor运行名: Actor-go2_sac_walking_training")
print("   Learner运行名: Learner-go2_sac_walking_training")
print()
print("🌐 WandB链接: https://wandb.ai/your-username/go2_sac_training")
print("=" * 50)
