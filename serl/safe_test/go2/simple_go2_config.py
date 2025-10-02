#!/usr/bin/env python3

"""    class control:
        action            # 限制约束
            dof_pos_limits = -1.0         # 减轻：关节位置限制
            collision = -1.0              # 增强：碰撞惩罚
            termination = -2.0            # 启用：早期终止惩罚le = 0.25  # GO2 original action scale
        decimation = 4plified GO2 configuration for testing when go2_rl module is not available"""

class SimpleGO2Config:
    class init_state:
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {
            # Angles EXACTLY matching deploy_mujoco go2.yaml and go2_config.py
            'FL_hip_joint': 0.1,    # [rad] - from deploy_mujoco [0.1, 0.8, -1.5]
            'FR_hip_joint': -0.1,   # [rad] - from deploy_mujoco [-0.1, 0.8, -1.5]
            'RL_hip_joint': 0.1,    # [rad] - from deploy_mujoco [0.1, 1.0, -1.5]
            'RR_hip_joint': -0.1,   # [rad] - from deploy_mujoco [-0.1, 1.0, -1.5]
            'FL_thigh_joint': 0.8,  # [rad] - matching deploy_mujoco and go2_config.py
            'FR_thigh_joint': 0.8,  # [rad] - matching deploy_mujoco and go2_config.py
            'RL_thigh_joint': 1.0,  # [rad] - matching deploy_mujoco and go2_config.py
            'RR_thigh_joint': 1.0,  # [rad] - matching deploy_mujoco and go2_config.py
            'FL_calf_joint': -1.5,  # [rad] - matching deploy_mujoco and go2_config.py
            'FR_calf_joint': -1.5,  # [rad] - matching deploy_mujoco and go2_config.py
            'RL_calf_joint': -1.5,  # [rad] - matching deploy_mujoco and go2_config.py
            'RR_calf_joint': -1.5,  # [rad] - matching deploy_mujoco and go2_config.py
        }
    
    class env:
        num_observations = 48
        enable_early_termination = True  # 启用早期终止以加快训练
        use_sparse_reward = False  # 设置为True使用稀疏奖励，False使用复杂奖励
    
    class control:
        action_scale = 0.25  # Reduced from 0.25 to prevent large joint movements
        decimation = 4
    
    class sim:
        dt = 0.002
    
    class rewards:
        base_height_target = 0.34
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        
        class scales:
            # 正向奖励 - 鼓励前进和存活
            alive = 2.0                    # 新增：存活奖励
            tracking = 1.0                 # 新增：前进奖励
            tracking_lin_vel = 2.0         # 增强：线速度跟踪
            tracking_ang_vel = -2.5         # 保持：角速度跟踪
            
            # 姿态和稳定性
            orientation = -2.0             # 增强：防止翻倒
            base_height = -0.5             # 新增：高度保持
            lin_vel_z = -1.0              
            ang_vel_xy = -0.1             # 增强：防止侧翻
            # 动作平滑性
            action_rate = -0.01           # 保持：动作变化率
            dof_vel = -0.0              # 增强：关节速度惩罚
            dof_acc = -2.5e-7               # 增强：关节加速度惩罚
            torques = -0.0001             # 保持：力矩惩罚
            
            # 限制约束
            dof_pos_limits = -0.0         # 减轻：关节位置限制
            collision = -0.1              # 增强：碰撞惩罚
            termination = -10.0           # 增强：终止惩罚
            
            # 暂时关闭的奖励
            stand_still = 0.0
            feet_air_time = 0.0
            feet_stumble = 0.0
