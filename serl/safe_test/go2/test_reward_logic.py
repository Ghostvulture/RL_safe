#!/usr/bin/env python3
"""
简化的奖励系统测试 - 验证逻辑改进
"""

import numpy as np

class MockRewardTester:
    """模拟奖励测试器"""
    
    def __init__(self):
        self.dt = 0.02
    
    def _tolerance_reward(self, x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
        """简化的tolerance函数"""
        lower, upper = bounds
        
        in_bounds = (lower <= x <= upper)
        if margin == 0:
            return 1.0 if in_bounds else 0.0
        else:
            if in_bounds:
                return 1.0
            else:
                d = min(abs(x - lower), abs(x - upper)) / margin
                if sigmoid == 'linear':
                    return max(0.0, 1.0 - d + value_at_margin * d)
                elif sigmoid == 'gaussian':
                    return value_at_margin * np.exp(-0.5 * d**2)
                else:
                    return value_at_margin * np.exp(-d)
    
    def test_improved_sparse_reward(self, lin_vel, pitch, roll, height, base_ang_vel):
        """测试改进后的稀疏奖励"""
        
        target_vel = 0.5
        
        # 1. 更严格的速度奖励
        adjusted_vel = lin_vel * np.cos(pitch)
        
        if adjusted_vel < 0.1:  # 速度太低
            velocity_reward = -1.0
        elif adjusted_vel < target_vel * 0.5:  # 速度不足
            velocity_reward = self._tolerance_reward(
                x=adjusted_vel,
                bounds=(0.1, target_vel * 0.5),
                margin=0.2,
                value_at_margin=0.0,
                sigmoid='linear'
            ) - 0.5
        else:  # 速度较好
            velocity_reward = self._tolerance_reward(
                x=adjusted_vel,
                bounds=(target_vel * 0.5, target_vel * 1.5),
                margin=target_vel,
                value_at_margin=0.2,
                sigmoid='gaussian'
            )
        
        # 2. 增强的倾倒惩罚
        tilt_penalty = 0.0
        if abs(pitch) > 0.3:
            tilt_penalty += 2.0 * (abs(pitch) - 0.3)
        if abs(roll) > 0.3:
            tilt_penalty += 1.0 * (abs(roll) - 0.3)
        
        # 3. Yaw惩罚
        yaw_rate_penalty = 0.2 * abs(base_ang_vel[2]) if len(base_ang_vel) > 2 else 0.0
        
        # 4. 稳定性奖励
        height_stability = self._tolerance_reward(
            x=height, bounds=(0.25, 0.45), margin=0.15, sigmoid='gaussian'
        )
        roll_stability = self._tolerance_reward(
            x=abs(roll), bounds=(0.0, 0.2), margin=0.3, sigmoid='gaussian'
        )
        pitch_stability = self._tolerance_reward(
            x=abs(pitch), bounds=(0.0, 0.15), margin=0.25, sigmoid='gaussian'
        )
        stability_reward = (height_stability * 0.3 + roll_stability * 0.25 + pitch_stability * 0.25)
        
        # 与速度相关的稳定性权重
        if adjusted_vel < 0.2:
            stability_reward *= 0.3
        
        # 5. 接触因子
        height_ok = 0.2 < height < 0.5
        orientation_ok = abs(roll) < 0.4 and abs(pitch) < 0.4
        angular_vel_ok = np.linalg.norm(base_ang_vel) < 2.0
        
        is_falling = (abs(pitch) > 0.35 or abs(roll) > 0.35 or 
                     height < 0.18 or
                     np.linalg.norm(base_ang_vel[:2]) > 1.5)
        
        if is_falling:
            contact_factor = 0.05
        elif height_ok and orientation_ok and angular_vel_ok:
            contact_factor = 0.8
        else:
            contact_factor = 0.2
        
        # 奖励组合
        base_reward = velocity_reward - yaw_rate_penalty - tilt_penalty
        
        if base_reward > 0:
            bonus_rewards = stability_reward * 0.5
            total_reward = (base_reward + bonus_rewards) * contact_factor * 10.0
        else:
            total_reward = base_reward * contact_factor * 10.0
        
        return {
            'velocity_reward': velocity_reward,
            'tilt_penalty': tilt_penalty,
            'yaw_penalty': yaw_rate_penalty,
            'stability_reward': stability_reward,
            'contact_factor': contact_factor,
            'base_reward': base_reward,
            'total_reward': total_reward,
            'adjusted_vel': adjusted_vel
        }

def test_scenarios():
    """测试不同场景"""
    
    tester = MockRewardTester()
    
    print("=== 改进后的稀疏奖励测试 ===\n")
    
    scenarios = [
        {
            'name': '正常站立，无前进',
            'lin_vel': 0.001,
            'pitch': 0.0,
            'roll': 0.0,
            'height': 0.34,
            'base_ang_vel': [0.0, 0.0, 0.0]
        },
        {
            'name': '向前倾倒状态',
            'lin_vel': 0.001,
            'pitch': 0.4,  # 约23度
            'roll': 0.0,
            'height': 0.25,
            'base_ang_vel': [0.0, 0.5, 0.0]
        },
        {
            'name': '正常前进',
            'lin_vel': 0.6,
            'pitch': 0.0,
            'roll': 0.0,
            'height': 0.34,
            'base_ang_vel': [0.0, 0.0, 0.0]
        },
        {
            'name': '严重倾倒',
            'lin_vel': 0.0,
            'pitch': 0.6,  # 约34度
            'roll': 0.0,
            'height': 0.15,
            'base_ang_vel': [0.0, 1.0, 0.0]
        },
        {
            'name': '低速前进但stable',
            'lin_vel': 0.3,
            'pitch': 0.0,
            'roll': 0.0,
            'height': 0.34,
            'base_ang_vel': [0.0, 0.0, 0.0]
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        result = tester.test_improved_sparse_reward(
            scenario['lin_vel'],
            scenario['pitch'],
            scenario['roll'],
            scenario['height'],
            scenario['base_ang_vel']
        )
        results.append((scenario['name'], result))
        
        print(f"场景: {scenario['name']}")
        print(f"  速度: {scenario['lin_vel']:.3f} m/s, 调整后: {result['adjusted_vel']:.3f}")
        print(f"  速度奖励: {result['velocity_reward']:.3f}")
        print(f"  倾倒惩罚: {result['tilt_penalty']:.3f}")
        print(f"  稳定性奖励: {result['stability_reward']:.3f}")
        print(f"  接触因子: {result['contact_factor']:.3f}")
        print(f"  基础奖励: {result['base_reward']:.3f}")
        print(f"  总奖励: {result['total_reward']:.3f}")
        print()
    
    print("=== 改进效果验证 ===")
    rewards = [r[1]['total_reward'] for r in results]
    names = [r[0] for r in results]
    
    print(f"各场景奖励对比:")
    for name, reward in zip(names, rewards):
        print(f"  {name:15}: {reward:8.3f}")
    
    print(f"\n改进验证:")
    print(f"✓ 倾倒状态比站立更负:     {rewards[1] < rewards[0]}")
    print(f"✓ 严重倾倒奖励最负:       {rewards[3] == min(rewards)}")
    print(f"✓ 正常前进奖励最高:       {rewards[2] == max(rewards)}")
    print(f"✓ 无前进速度奖励足够低:    {rewards[0] < 1.0}")
    print(f"✓ 倾倒状态奖励为负:       {rewards[1] < 0 and rewards[3] < 0}")

if __name__ == "__main__":
    test_scenarios()
