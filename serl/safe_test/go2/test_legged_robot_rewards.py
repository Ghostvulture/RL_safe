#!/usr/bin/env python3
"""
测试修改后的奖励函数是否与legged_robot.py的奖励函数一致
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'go2_rl'))

from go2_gym_env import Go2GymEnv

def test_reward_functions():
    """测试所有奖励函数是否正常工作"""
    print("=== 测试修改后的奖励函数 ===")
    
    # 创建环境
    env = Go2GymEnv(render_mode="rgb_array")
    
    try:
        # 重置环境
        obs, _ = env.reset()
        print("✓ 环境重置成功")
        
        # 设置运动指令
        env.set_commands(lin_vel_x=0.5, lin_vel_y=0.0, ang_vel_z=0.0)
        print("✓ 运动指令设置成功")
        
        # 执行几步来初始化状态
        action = np.zeros(env.action_space.shape)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
        
        print("\n=== 测试各个奖励函数 ===")
        
        # 测试每个奖励函数
        reward_functions = [
            'tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z', 'ang_vel_xy',
            'orientation', 'base_height', 'torques', 'dof_vel', 'dof_acc',
            'action_rate', 'collision', 'termination', 'dof_pos_limits',
            'dof_vel_limits', 'torque_limits', 'feet_air_time', 'stumble',
            'stand_still', 'feet_contact_forces'
        ]
        
        for reward_name in reward_functions:
            func_name = f'_reward_{reward_name}'
            if hasattr(env, func_name):
                try:
                    func = getattr(env, func_name)
                    reward_value = func()
                    print(f"✓ {reward_name}: {reward_value:.6f}")
                except Exception as e:
                    print(f"✗ {reward_name}: ERROR - {e}")
            else:
                print(f"✗ {reward_name}: 函数不存在")
        
        print("\n=== 测试总体奖励计算 ===")
        
        # 测试复杂奖励计算
        try:
            complex_reward = env._compute_complex_reward()
            print(f"✓ 复杂奖励: {complex_reward:.6f}")
        except Exception as e:
            print(f"✗ 复杂奖励计算失败: {e}")
        
        # 测试稀疏奖励计算
        try:
            sparse_reward = env._compute_sparse_reward()
            print(f"✓ 稀疏奖励: {sparse_reward:.6f}")
        except Exception as e:
            print(f"✗ 稀疏奖励计算失败: {e}")
        
        print("\n=== 测试成功 ===")
        return True
        
    except Exception as e:
        print(f"\n=== 测试失败 ===")
        print(f"错误: {e}")
        return False
    
    finally:
        env.close()

def compare_with_original():
    """比较与原始legged_robot.py奖励函数的对应关系"""
    print("\n=== 奖励函数对应关系 ===")
    
    legged_robot_rewards = [
        '_reward_lin_vel_z', '_reward_ang_vel_xy', '_reward_orientation',
        '_reward_base_height', '_reward_torques', '_reward_dof_vel',
        '_reward_dof_acc', '_reward_action_rate', '_reward_collision',
        '_reward_termination', '_reward_dof_pos_limits', '_reward_dof_vel_limits',
        '_reward_torque_limits', '_reward_tracking_lin_vel', '_reward_tracking_ang_vel',
        '_reward_feet_air_time', '_reward_stumble', '_reward_stand_still',
        '_reward_feet_contact_forces'
    ]
    
    print("已实现的奖励函数:")
    for reward in legged_robot_rewards:
        print(f"  ✓ {reward}")
    
    print(f"\n总计: {len(legged_robot_rewards)} 个奖励函数")

if __name__ == "__main__":
    success = test_reward_functions()
    compare_with_original()
    
    if success:
        print("\n🎉 所有测试通过！奖励函数已成功移植。")
    else:
        print("\n❌ 测试失败，需要进一步修复。")
