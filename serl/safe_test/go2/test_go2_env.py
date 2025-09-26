#!/usr/bin/env python3

"""Test script for GO2 Gym Environment"""

import numpy as np
import sys
import os

# Add the path to import go2_gym_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from go2_gym_env import Go2GymEnv
    print("Successfully imported Go2GymEnv")
except ImportError as e:
    print(f"Failed to import Go2GymEnv: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic environment functionality."""
    try:
        # Create environment
        print("Creating Go2GymEnv...")
        env = Go2GymEnv(render_mode="rgb_array")
        print(f"Action space: {env.action_space}")
        print(f"Observation space keys: {list(env.observation_space['state'].spaces.keys())}")
        
        # Test reset
        print("Testing reset...")
        obs, info = env.reset()
        print(f"Reset successful. Observation keys: {list(obs['state'].keys())}")
        
        # Test a few steps
        print("Testing steps...")
        for i in range(5):
            action = np.random.uniform(-1, 1, 12)  # 12 actions for GO2
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward = {reward:.4f}, terminated = {terminated}")
            
            if terminated:
                print("Episode terminated early")
                break
        
        print("Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_functions():
    """Test reward function computation."""
    try:
        print("Testing reward functions...")
        env = Go2GymEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Test reward computation
        action = np.zeros(12)  # Zero action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward with zero action: {reward:.4f}")
        
        # Test with random actions
        for i in range(3):
            action = np.random.uniform(-0.5, 0.5, 12)  # Small random actions
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward with random action {i+1}: {reward:.4f}")
        
        print("Reward function test passed!")
        return True
        
    except Exception as e:
        print(f"Reward test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting GO2 Environment Tests")
    print("=" * 50)
    
    # Test basic functionality
    basic_test_passed = test_basic_functionality()
    print()
    
    # Test reward functions
    reward_test_passed = test_reward_functions()
    print()
    
    # Summary
    print("Test Summary:")
    print(f"Basic functionality: {'PASSED' if basic_test_passed else 'FAILED'}")
    print(f"Reward functions: {'PASSED' if reward_test_passed else 'FAILED'}")
    
    if basic_test_passed and reward_test_passed:
        print("\nAll tests passed! GO2 environment is working correctly.")
    else:
        print("\nSome tests failed. Please check the errors above.")
