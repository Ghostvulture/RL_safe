#!/usr/bin/env python3

# Test basic SAC agent creation with GO2 environment
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'safe_test', 'go2'))

# Import required modules
import jax
import numpy as np
from go2_gym_env import Go2GymEnv
import gym
from serl_launcher.utils.launcher import make_sac_agent

def flatten_observation(obs_dict):
    obs_list = []
    for key in sorted(obs_dict.keys()):
        if isinstance(obs_dict[key], dict):
            for subkey in sorted(obs_dict[key].keys()):
                obs_list.append(obs_dict[key][subkey].flatten())
        else:
            obs_list.append(obs_dict[key].flatten())
    return np.concatenate(obs_list)

class FlattenedGO2Env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
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
    env = Go2GymEnv(render_mode=render_mode)
    env = FlattenedGO2Env(env)
    return env

def test_sac_agent_creation():
    print("Testing SAC agent creation with GO2 environment...")
    
    # Create environment
    env = make_go2_env()
    print(f"Environment created: Action space {env.action_space}, Obs space {env.observation_space}")
    
    # Create SAC agent
    try:
        agent = make_sac_agent(
            seed=42,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
        )
        print("SAC agent created successfully!")
        
        # Test agent action sampling
        obs, _ = env.reset()
        rng = jax.random.PRNGKey(42)
        action = agent.sample_actions(
            observations=jax.device_put(obs),
            seed=rng,
            deterministic=False,
        )
        action = np.asarray(jax.device_get(action))
        print(f"Agent action sampling successful: {action.shape}")
        
        # Test environment step
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"Environment step successful: reward={reward}, done={done}")
        
        print("All SAC integration tests passed!")
        
    except Exception as e:
        print(f"Error creating SAC agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sac_agent_creation()
