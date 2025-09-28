#!/usr/bin/env python3

"""
优化后的GO2配置文件，专门用于从零开始学习行走
关键改进：
1. 增加正向奖励激励
2. 减少过度惩罚
3. 更好的奖励平衡
"""

class OptimizedGO2Config:
    """优化的GO2训练配置"""
    
    class init_state:
        pos = [0.0, 0.0, 0.34]  # x,y,z [m] - GO2站立高度
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        
        # GO2默认关节角度（站立姿态）
        default_joint_angles = {
            'FL_hip_joint': 0.1,     'FL_thigh_joint': 0.8,   'FL_calf_joint': -1.5,
            'FR_hip_joint': -0.1,    'FR_thigh_joint': 0.8,   'FR_calf_joint': -1.5,
            'RL_hip_joint': 0.1,     'RL_thigh_joint': 1.0,   'RL_calf_joint': -1.5,
            'RR_hip_joint': -0.1,    'RR_thigh_joint': 1.0,   'RR_calf_joint': -1.5,
        }

    class control:
        control_type = 'P'  # Position control
        stiffness = {'joint': 20.0}  # [N*m/rad] 
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25          # Smaller scale for smoother actions
        decimation = 4               # Control frequency

    class rewards:
        base_height_target = 0.34    # GO2目标高度
        tracking_sigma = 0.25        # 跟踪奖励的平滑参数
        soft_dof_pos_limit = 0.9
        
        class scales:
            # 🎯 核心正向奖励 - 鼓励基本行为
            alive = 2.0                    # 存活奖励 - 鼓励站立
            tracking_lin_vel = 1.5         # 线速度跟踪 - 鼓励向前走
            tracking_ang_vel = 0.5         # 角速度跟踪
            base_height = -0.5             # 高度保持（轻微惩罚偏离）
            
            # 🚫 基本约束 - 防止危险行为
            orientation = -1.0             # 姿态保持 - 防止翻倒
            lin_vel_z = -2.0              # 防止垂直跳跃
            ang_vel_xy = -0.1             # 防止侧翻
            collision = -5.0              # 碰撞惩罚
            
            # ⚡ 动作平滑性 - 鼓励稳定控制
            action_rate = -0.01           # 动作变化率
            dof_vel = -0.001              # 关节速度惩罚（轻微）
            dof_acc = -1e-6               # 关节加速度惩罚（很轻微）
            torques = -0.0001             # 力矩惩罚（轻微）
            
            # 📏 关节限制 - 防止过度运动
            dof_pos_limits = -1.0         # 关节位置限制（减轻惩罚）
            dof_vel_limits = -0.1         # 关节速度限制
            
            # 🦶 步态相关（先设为0，后期可开启）
            feet_air_time = 0.0          # 空中时间奖励
            feet_contact_forces = 0.0    # 接触力奖励
            stumble = -0.1               # 绊倒惩罚
            
            # 🎯 终止惩罚
            termination = -10.0          # 终止惩罚（适中）

    class commands:
        # 简化命令，专注于前进
        num_commands = 3
        resampling_time = 5.0  # 命令切换时间
        
        class ranges:
            lin_vel_x = [0.0, 0.8]    # 只向前走，从0到0.8m/s
            lin_vel_y = [0.0, 0.0]    # 不侧向移动
            ang_vel_yaw = [-0.2, 0.2] # 小幅转向

    class curriculum:
        # 课程学习设置
        base_height_target_min = 0.30     # 最低高度目标
        base_height_target_max = 0.38     # 最高高度目标
        lin_vel_x_max_start = 0.2         # 起始最大速度
        lin_vel_x_max_end = 0.8           # 最终最大速度

def get_optimized_scales():
    """返回优化的奖励权重"""
    return {
        # 正向激励
        'alive': 2.0,
        'tracking_lin_vel': 1.5,
        'tracking_ang_vel': 0.5,
        
        # 基本约束
        'orientation': -1.0,
        'base_height': -0.5,
        'lin_vel_z': -2.0,
        'ang_vel_xy': -0.1,
        'collision': -5.0,
        
        # 平滑控制
        'action_rate': -0.01,
        'dof_vel': -0.001,
        'dof_acc': -1e-6,
        'torques': -0.0001,
        
        # 限制约束
        'dof_pos_limits': -1.0,
        'dof_vel_limits': -0.1,
        'stumble': -0.1,
        
        # 终止
        'termination': -10.0
    }

def print_reward_analysis():
    """分析奖励设置"""
    scales = get_optimized_scales()
    
    print("🎯 GO2优化奖励分析")
    print("=" * 50)
    
    positive_rewards = {k: v for k, v in scales.items() if v > 0}
    negative_rewards = {k: v for k, v in scales.items() if v < 0}
    
    print(f"✅ 正向奖励 ({len(positive_rewards)} 项):")
    for name, scale in positive_rewards.items():
        print(f"  • {name}: +{scale}")
    
    print(f"\n❌ 负向惩罚 ({len(negative_rewards)} 项):")
    for name, scale in negative_rewards.items():
        print(f"  • {name}: {scale}")
    
    total_positive = sum(positive_rewards.values())
    total_negative = sum(abs(v) for v in negative_rewards.values())
    
    print(f"\n📊 奖励平衡:")
    print(f"  正向权重总和: +{total_positive:.1f}")
    print(f"  负向权重总和: -{total_negative:.1f}")
    print(f"  平衡比例: {total_positive/total_negative:.2f}:1")

if __name__ == "__main__":
    print_reward_analysis()
