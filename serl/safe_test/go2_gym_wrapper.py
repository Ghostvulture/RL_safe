import gym
from gym.spaces import flatten_space, flatten
import numpy as np
from typing import Callable, Union
from go2_rl.legged_gym.envs.base import legged_robot
from go2_rl.legged_gym.envs.go2.go2_config import GO2RoughCfg

from isaacgym import gymapi
from isaacgym.torch_utils import *


class Go2Wrapper(gym.Wrapper):
    """
    Goal-conditioned wrapper for Go2Robot environment.
    
    This wrapper handles the observation and removes the image processing part.
    Instead, it only returns the state space.
    """
    
    def __init__(self, 
                 env: Go2Robot, 
                 cfg: GO2RoughCfg,
                 goal_sampler: Union[np.ndarray, Callable]
                 ):
        super().__init__(env)
        self.env = env
        self.goal_sampler = goal_sampler
        self.current_goal = None

        num_observations = self.cfg.env.num_observations
        
        # 修改观察空间，仅保留状态信息，不处理图像
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.env.observation_space["state"]),  # 扁平化状态空间
                "images": gym.spaces.Box(low=np.zeros((128, 128, 3)), high=np.ones((128, 128, 3)), dtype=np.uint8),  # 占位符
            }
        )

    def step(self, *args):
        """
        执行一步操作，并返回转换后的观察、奖励、结束标志等信息
        """
        obs, reward, done, trunc, info = self.env.step(*args)  #TODO: no step in env!!!
        
        # 只处理状态信息，不涉及图像
        # obs = {
        #     "state": flatten(self.env.observation_space["state"], obs["state"]),  # 扁平化状态空间
        #     "images": np.zeros((128, 128, 3), dtype=np.uint8),  # 图像部分填充空值
        # }
        
        obs_dict = {
            "state": obs["state"],  # 假设state已经是一个合适的向量
            "images": np.zeros((128, 128, 3), dtype=np.uint8),  # 填充图像占位符
        }

        # # 过滤掉不需要的信息
        # info = filter_info_keys_go2(info)
        
        # 返回目标以及相关信息
        return obs_dict, reward, done, trunc, {"goal": self.current_goal, **info}
    
    def reset(self, **kwargs):
        """
        重置环境，采样一个新的目标
        """
        obs_dict = self.env.reset()
        return obs_dict, {}
