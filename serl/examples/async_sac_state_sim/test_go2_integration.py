#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'safe_test', 'go2'))

from go2_gym_env import Go2GymEnv
import gym

def flatten_observation(obs_dict):
    # Concatenate all observation components
    obs_list = []
    for key in sorted(obs_dict.keys()):  # Sort keys for consistent ordering
        if isinstance(obs_dict[key], dict):
            # Handle nested dict (state)
            for subkey in sorted(obs_dict[key].keys()):
                obs_list.append(obs_dict[key][subkey].flatten())
        else:
            obs_list.append(obs_dict[key].flatten())
    return np.concatenate(obs_list)

class FlattenedGO2Env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate flattened observation space size
        sample_obs = env.observation_space.sample()
        flattened_obs = flatten_observation(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flattened_obs.shape, dtype=np.float32
        )
        
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return flatten_observation(obs_dict), info
        
    def step(self, action):
        obs_dict, reward, done, truncated, info = self.env.step(action)
        return flatten_observation(obs_dict), reward, done, truncated, info

def make_go2_env(render_mode="rgb_array"):
    """Create GO2 gym environment"""
    env = Go2GymEnv(render_mode=render_mode)
    env = FlattenedGO2Env(env)
    return env

def test_env_creation():
    print("Testing GO2 environment creation...")
    
    # Test environment creation
    env = make_go2_env()
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Reset successful. Observation shape: {obs.shape}")
    
    # Test step
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"Step successful. Reward: {reward}, Done: {done}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_env_creation()
