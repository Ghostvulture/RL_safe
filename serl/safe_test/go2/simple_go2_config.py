#!/usr/bin/env python3

"""Simplified GO2 configuration for testing when go2_rl module is not available"""

class SimpleGO2Config:
    class init_state:
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RL_hip_joint': 0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }
    
    class env:
        num_observations = 48
    
    class control:
        action_scale = 0.25
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
            torques = -0.0002
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = 0.0
            action_rate = -0.01
            dof_pos_limits = -10.0
            stand_still = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            termination = 0.0
