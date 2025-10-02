#!/usr/bin/env python3
"""
测试改进后的稀疏奖励系统 - 验证向前倾倒时的惩罚效果
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'go2_rl'))

from go2_gym_env import Go2GymEnv

def test_reward_scenarios():
    """测试不同场景下的奖励计算"""
    
    # 创建环境
    env = Go2GymEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    
    print("=== 测试改进后的稀疏奖励系统 ===\n")
    
    # 测试场景1：正常站立，无前进
    print("场景1：正常站立，无前进速度")
    env._data.qvel[:3] = [0.001, 0.0, 0.0]  # 几乎无前进速度
    env._data.qpos[2] = 0.34  # 正常高度
    env._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 正常姿态
    env._data.qvel[3:6] = [0.0, 0.0, 0.0]  # 无角速度
    reward1 = env._compute_sparse_reward()
    print(f"奖励: {reward1:.3f} (应该是负值或很小的正值)\n")
    
    # 测试场景2：向前倾倒状态
    print("场景2：向前倾倒状态")
    env._data.qvel[:3] = [0.001, 0.0, 0.0]  # 几乎无前进速度
    env._data.qpos[2] = 0.25  # 较低高度
    # 模拟向前倾倒的姿态 (pitch = 0.4 rad ≈ 23度)
    import mujoco
    pitch_angle = 0.4
    env._data.qpos[3:7] = [np.cos(pitch_angle/2), 0.0, np.sin(pitch_angle/2), 0.0]
    env._data.qvel[3:6] = [0.0, 0.5, 0.0]  # 有pitch角速度
    reward2 = env._compute_sparse_reward()
    print(f"奖励: {reward2:.3f} (应该是明显的负值)\n")
    
    # 测试场景3：正常前进
    print("场景3：正常前进")
    env._data.qvel[:3] = [0.6, 0.0, 0.0]  # 良好的前进速度
    env._data.qpos[2] = 0.34  # 正常高度
    env._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 正常姿态
    env._data.qvel[3:6] = [0.0, 0.0, 0.0]  # 无角速度
    # 模拟hip关节运动
    env._data.qvel[6:18] = [1.0, 0.5, 0.2, -1.0, -0.5, -0.2,
                           -1.0, -0.5, -0.2, 1.0, 0.5, 0.2]  # 对角线步态
    reward3 = env._compute_sparse_reward()
    print(f"奖励: {reward3:.3f} (应该是正值)\n")
    
    # 测试场景4：严重倾倒
    print("场景4：严重倾倒")
    env._data.qvel[:3] = [0.0, 0.0, 0.0]  # 无前进速度
    env._data.qpos[2] = 0.15  # 很低的高度
    # 模拟严重向前倾倒 (pitch = 0.6 rad ≈ 34度)
    pitch_angle = 0.6
    env._data.qpos[3:7] = [np.cos(pitch_angle/2), 0.0, np.sin(pitch_angle/2), 0.0]
    env._data.qvel[3:6] = [0.0, 1.0, 0.0]  # 大的pitch角速度
    reward4 = env._compute_sparse_reward()
    print(f"奖励: {reward4:.3f} (应该是强烈的负值)\n")
    
    # 测试场景5：低速前进但stable
    print("场景5：低速前进但stable")
    env._data.qvel[:3] = [0.3, 0.0, 0.0]  # 中等前进速度
    env._data.qpos[2] = 0.34  # 正常高度
    env._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 正常姿态
    env._data.qvel[3:6] = [0.0, 0.0, 0.0]  # 无角速度
    reward5 = env._compute_sparse_reward()
    print(f"奖励: {reward5:.3f} (应该是中等正值)\n")
    
    print("=== 奖励测试完成 ===")
    print(f"场景对比：")
    print(f"正常站立无前进: {reward1:.3f}")
    print(f"向前倾倒:       {reward2:.3f}")
    print(f"正常前进:       {reward3:.3f}")
    print(f"严重倾倒:       {reward4:.3f}")
    print(f"低速前进:       {reward5:.3f}")
    
    print(f"\n改进效果验证:")
    print(f"✓ 倾倒状态奖励是否比站立更负: {reward2 < reward1}")
    print(f"✓ 严重倾倒奖励是否最负:     {reward4 < reward2}")
    print(f"✓ 正常前进奖励是否最高:     {reward3 > max(reward1, reward2, reward4, reward5)}")
    print(f"✓ 无前进速度奖励是否足够低:  {reward1 < 5.0}")
    
    env.close()

def test_component_analysis():
    """分析各个奖励组件的贡献"""
    
    env = Go2GymEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    
    print("\n=== 奖励组件分析 ===")
    
    # 设置一个典型的"倾倒但stability高"的状态
    env._data.qvel[:3] = [0.001, 0.0, 0.0]  # 几乎无前进
    env._data.qpos[2] = 0.34  # 正常高度
    env._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 正常姿态
    env._data.qvel[3:6] = [0.0, 0.0, 0.0]  # 无角速度
    
    # 更新状态
    env._update_reward_state()
    
    # 计算各个组件
    roll, pitch, _ = env._get_euler_from_quat(env.base_quat)
    lin_vel = env.base_lin_vel[0]
    adjusted_vel = lin_vel * np.cos(pitch)
    
    # 速度奖励
    if adjusted_vel < 0.1:
        velocity_reward = -1.0
    else:
        velocity_reward = env._tolerance_reward(
            x=adjusted_vel,
            bounds=(0.1, 0.25),
            margin=0.2,
            value_at_margin=0.0,
            sigmoid='linear'
        ) - 0.5
    
    hip_movement_reward = env._compute_hip_movement_reward()
    stability_reward = env._compute_stability_reward()
    contact_factor = env._compute_contact_factor()
    
    print(f"速度: {lin_vel:.3f} m/s, 调整后速度: {adjusted_vel:.3f}")
    print(f"速度奖励: {velocity_reward:.3f}")
    print(f"Hip运动奖励: {hip_movement_reward:.3f}")
    print(f"稳定性奖励: {stability_reward:.3f}")
    print(f"接触因子: {contact_factor:.3f}")
    print(f"姿态 - roll: {roll:.3f}, pitch: {pitch:.3f}")
    
    total = env._compute_sparse_reward()
    print(f"总奖励: {total:.3f}")
    
    env.close()

if __name__ == "__main__":
    test_reward_scenarios()
    test_component_analysis()
