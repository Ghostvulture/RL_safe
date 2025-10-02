#!/usr/bin/env python3

"""
Sparse GO2 configuration for SAC fine-tuning style training.
Based on simple_go2_config.py but with sparse reward enabled.
"""

from simple_go2_config import SimpleGO2Config

class SparseGO2Config(SimpleGO2Config):
    """GO2 config with sparse reward for SAC fine-tuning style training."""
    
    class env(SimpleGO2Config.env):
        use_sparse_reward = True  # 启用稀疏奖励模式
        enable_early_termination = True  # 保持早期终止
        
    # 稀疏奖励模式下不需要复杂的奖励权重，但保留原始结构以防需要
    class rewards(SimpleGO2Config.rewards):
        # 在稀疏模式下，这些权重不会被使用
        # 但保留结构以便在调试时可以快速切换回复杂奖励
        class scales(SimpleGO2Config.rewards.scales):
            # 所有奖励权重设为0，因为使用稀疏奖励
            alive = 0.0
            tracking = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            orientation = 0.0
            base_height = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            action_rate = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            torques = 0.0
            dof_pos_limits = 0.0
            collision = 0.0
            termination = 0.0
            stand_still = 0.0
            feet_air_time = 0.0
            feet_stumble = 0.0
