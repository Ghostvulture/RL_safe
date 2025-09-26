# from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
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


# register(
#     id="PandaPickCube-v0",
#     entry_point="franka_sim.envs:PandaPickCubeGymEnv",
#     max_episode_steps=100,
# )
# register(
#     id="PandaPickCubeVision-v0",
#     entry_point="franka_sim.envs:PandaPickCubeGymEnv",
#     max_episode_steps=100,
#     kwargs={"image_obs": True},
# )