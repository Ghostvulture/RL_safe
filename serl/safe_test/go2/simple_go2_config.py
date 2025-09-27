#!/usr/bin/env python3

"""    class control:
        action_scale = 0.25  # GO2 original action scale
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
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = 0.0
            torques = -0.0001
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = 0.0
            action_rate = -0.01
            dof_pos_limits = -10.0
            stand_still = 0.0
            feet_air_time = 0.0
            collision = -1.
            feet_stumble = 0.0
            termination = 0.0
