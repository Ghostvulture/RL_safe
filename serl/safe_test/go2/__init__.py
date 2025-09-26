from safe_test.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register


register(
    id="unitree-go2",
    entry_point="safe_test.envs:Go2GymEnv",
    max_episode_steps=100,
)

