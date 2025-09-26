from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np
import sys
import os

# Add the path to import go2_rl modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

#import from legged_gym base
try:
    from go2_rl.legged_gym.envs.base.legged_robot import LeggedRobot
    from go2_rl.legged_gym.envs.go2.go2_config import GO2RoughCfg
except ImportError:
    print("Warning: go2_rl module not found, using simplified config")
    from simple_go2_config import SimpleGO2Config as GO2RoughCfg
    LeggedRobot = None

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from controllers import opspace
from mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
# Path: /home/xyz/Desktop/xluo/RL_safe/serl/safe_test/go2 -> /home/xyz/Desktop/xluo/RL_safe
_XML_PATH = Path(__file__).parent / "go2_model" / "go2_fixed.xml"


class Go2GymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: float = GO2RoughCfg.control.action_scale,
        seed: int = 0,
        control_dt: float = 0.02,#TODO:控制频率
        physics_dt: float = GO2RoughCfg.sim.dt,#0.002
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,#TODO: change to go2 xml
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.render_width = render_spec.width if render_spec.width is not None else 1024  # Increased resolution
        self.render_height = render_spec.height if render_spec.height is not None else 768  # Increased resolution

        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Get joint names from config (these have _joint suffix)
        self.joint_names = list(GO2RoughCfg.init_state.default_joint_angles.keys())
        self.default_joint_angles = np.asarray([
            GO2RoughCfg.init_state.default_joint_angles[joint_name] 
            for joint_name in self.joint_names
        ])
        
        # Create actuator names (remove _joint suffix for actuator mapping)
        self.actuator_names = [joint_name.replace('_joint', '') for joint_name in self.joint_names]
        
        # Caching.
        self._joint_ids = np.asarray(
            [self._model.joint(joint_name).id for joint_name in self.joint_names]
        )
        self._actuator_ctrl_ids = np.asarray(
            [self._model.actuator(actuator_name).id for actuator_name in self.actuator_names]
        )

        self.num_actions = len(self.joint_names)  # 12 joints
        
        # GO2 observation structure: [vel(3, implementby Kalman@TODO)base_lin_vel(3), projected_gravity(3), commands(3), 
        #                            dof_pos(12), dof_vel(12), previous_actions(12)]
        self.num_observations = GO2RoughCfg.env.num_observations  # 48

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        # GO2 base velocity (3)
                        "go2/lin_vel": gym.spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        # GO2 base angular velocity (3)
                        "go2/ang_vel": gym.spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        # Projected gravity vector (3)
                        "go2/gravity_orientation": gym.spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        # Movement commands [lin_vel_x, lin_vel_y, ang_vel_z] (3)
                        "go2/cmd": gym.spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        # Joint positions (12)
                        "go2/joint_pos": gym.spaces.Box(
                            -np.inf, np.inf, shape=(self.num_actions,), dtype=np.float32
                        ),
                        # Joint velocities (12)
                        "go2/joint_vel": gym.spaces.Box(
                            -np.inf, np.inf, shape=(self.num_actions,), dtype=np.float32
                        ),
                        # Previous actions (12)
                        "go2/previous_actions": gym.spaces.Box(
                            -1.0, 1.0, shape=(self.num_actions,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        # GO2 action space: 12 joint position targets (normalized to [-1, 1])
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),  # 12 joints
            dtype=np.float32,
        )

        # Initialize GO2-specific variables
        self._previous_actions = np.zeros(self.num_actions, dtype=np.float32)
        self._commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        
        # Initialize reward system
        self._setup_reward_functions()

        # DEBUG: Print MuJoCo model info
        print(f"DEBUG: GO2 Model loaded successfully!")
        print(f"DEBUG: Model has {self._model.nlight} lights")
        print(f"DEBUG: Model has {self._model.ngeom} geometries")
        print(f"DEBUG: Render mode: {render_mode}")
        print(f"DEBUG: Render size: {self.render_width}x{self.render_height}")
        
        # Check if ground plane exists
        ground_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        print(f"DEBUG: Ground plane ID: {ground_geom_id}")
        
        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        try:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self._viewer = MujocoRenderer(
                self._model,
                self._data,
                height=self.render_height,
                width=self.render_width,
            )
            print("DEBUG: MujocoRenderer created successfully")
        except ImportError as e:
            print(f"Warning: MujocoRenderer not available: {e}, rendering disabled")
            self._viewer = None
        except Exception as e:
            print(f"ERROR: Failed to create MujocoRenderer: {e}")
            self._viewer = None
        
        if self.render_mode == "human" and self._viewer is not None:
            print("DEBUG: Attempting initial render...")
            try:
                self._viewer.render(self.render_mode)
                print("DEBUG: Initial render successful")
            except Exception as e:
                print(f"ERROR: Initial render failed: {e}")


    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the GO2 robot environment for walking training."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset GO2 to initial position (origin) and default joint positions
        # Set base position to initial state from config
        self._data.qpos[:3] = GO2RoughCfg.init_state.pos  # x, y, z position
        
        # Set base orientation (quaternion w, x, y, z)
        self._data.qpos[3:7] = GO2RoughCfg.init_state.rot  # quaternion
        
        # Reset joint positions to default angles
        self._data.qpos[self._joint_ids] = self.default_joint_angles
        
        # Reset velocities to zero
        self._data.qvel[:] = 0.0
        
        # Initialize previous actions to zero
        self._previous_actions = np.zeros(self.num_actions, dtype=np.float32)
        
        # Initialize movement commands (can be randomized for training diversity)
        self._commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        
        # Reset time
        self._time = 0.0
        
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the GO2 environment for walking training.
        Params:
            action: np.ndarray - 12 joint position targets (normalized [-1, 1])

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Store current action as previous action for next observation
        self._previous_actions = action.copy()
        
        # Convert normalized actions to joint targets
        # target_joint_pos = action_scale * action + default_joint_angles
        target_joint_pos = self._action_scale * action + self.default_joint_angles
        
        # Apply actions to actuators (position control)
        self._data.ctrl[self._actuator_ctrl_ids] = target_joint_pos
        
        # Step the physics simulation
        # Use decimation from config (default 4 steps per control step)
        decimation = getattr(GO2RoughCfg.control, 'decimation', 4)
        for _ in range(decimation):
            mujoco.mj_step(self._model, self._data)
        
        # Update time
        self._time += self.control_dt

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded()

        return obs, rew, terminated, False, {}

    def render(self):
        print(f"DEBUG: render() called, viewer exists: {self._viewer is not None}")
        if self._viewer is None:
            print("DEBUG: No viewer available, returning empty frames")
            return []
        
        print(f"DEBUG: Rendering with camera IDs: {self.camera_id}")
        rendered_frames = []
        for cam_id in self.camera_id:
            try:
                print(f"DEBUG: Rendering camera {cam_id}")
                frame = self._viewer.render(render_mode="rgb_array")#, camera_id=cam_id
                print(f"DEBUG: Frame shape: {frame.shape if hasattr(frame, 'shape') else 'N/A'}")
                rendered_frames.append(frame)
            except Exception as e:
                print(f"ERROR: Failed to render camera {cam_id}: {e}")
        
        print(f"DEBUG: Rendered {len(rendered_frames)} frames")
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        """Compute GO2 robot observations based on legged_gym structure."""
        obs = {}
        obs["state"] = {}

        # Get base body index (assuming base body is named "base")
        try:
            base_body_id = self._model.body("base").id
        except:
            # If "base" body doesn't exist, use root body (index 0)
            base_body_id = 0
        
        # Base linear velocity (3) - in world frame
        base_lin_vel = self._data.qvel[:3].copy()  # First 3 elements are base linear velocity
        obs["state"]["go2/lin_vel"] = base_lin_vel.astype(np.float32)
        
        # Base angular velocity (3) - in body frame  
        base_ang_vel = self._data.qvel[3:6].copy()  # Next 3 elements are base angular velocity
        obs["state"]["go2/ang_vel"] = base_ang_vel.astype(np.float32)
        
        # Projected gravity (3) - gravity vector in robot's body frame
        # Get base orientation quaternion
        base_quat = self._data.qpos[3:7].copy()  # [w, x, y, z] in MuJoCo
        # Convert to rotation matrix and project gravity
        gravity_world = np.array([0, 0, -1], dtype=np.float32)  # Gravity in world frame
        # For simplicity, approximate projected gravity (this should be computed properly with quaternion)
        projected_gravity = gravity_world  # Placeholder - should be rotated to body frame
        obs["state"]["go2/gravity_orientation"] = projected_gravity
        
        # Movement commands (3)
        obs["state"]["go2/cmd"] = self._commands.astype(np.float32)
        
        # Joint positions (12) - relative to default positions
        joint_positions = self._data.qpos[self._joint_ids].copy()
        joint_pos_relative = (joint_positions - self.default_joint_angles).astype(np.float32)
        obs["state"]["go2/joint_pos"] = joint_pos_relative
        
        # Joint velocities (12)
        # Joint velocities start after base DOFs (6 DOFs for base)
        joint_vel_ids = np.arange(6, 6 + len(self._joint_ids))
        joint_velocities = self._data.qvel[joint_vel_ids].copy().astype(np.float32)
        obs["state"]["go2/joint_vel"] = joint_velocities
        
        # Previous actions (12)
        obs["state"]["go2/previous_actions"] = self._previous_actions.astype(np.float32)

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.render(self.render_mode)

        return obs

    def _setup_reward_functions(self):
        """Setup reward functions based on GO2 config scales."""
        try:
            from go2_rl.legged_gym.utils.helpers import class_to_dict
            # Get reward scales from config
            self.reward_scales = class_to_dict(GO2RoughCfg.rewards.scales)
        except ImportError:
            # Fallback: manually extract scales from config
            self.reward_scales = {}
            scales_obj = GO2RoughCfg.rewards.scales
            for attr_name in dir(scales_obj):
                if not attr_name.startswith('_') and not callable(getattr(scales_obj, attr_name)):
                    self.reward_scales[attr_name] = getattr(scales_obj, attr_name)
        
        self.dt = self.control_dt  # Use control timestep
        
        # Remove zero scales and multiply by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        
        # Prepare list of reward functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            func_name = '_reward_' + name
            if hasattr(self, func_name):
                self.reward_functions.append(getattr(self, func_name))
            else:
                print(f"Warning: Reward function {func_name} not found, skipping.")

    def _compute_reward(self) -> float:
        """Compute rewards using GO2 reward functions."""
        total_reward = 0.0
        
        # Update state variables needed for reward computation
        self._update_reward_state()
        
        # Compute each reward component
        for i, reward_func in enumerate(self.reward_functions):
            name = self.reward_names[i]
            try:
                rew = reward_func() * self.reward_scales[name]
                total_reward += float(rew)
            except Exception as e:
                print(f"Error computing reward {name}: {e}")
                continue
        
        return float(total_reward)

    def _update_reward_state(self):
        """Update state variables needed for reward computation."""
        # Base position and velocities
        self.base_pos = self._data.qpos[:3].copy()
        self.base_quat = self._data.qpos[3:7].copy()  # [w, x, y, z] in MuJoCo
        self.base_lin_vel = self._data.qvel[:3].copy()
        self.base_ang_vel = self._data.qvel[3:6].copy()
        
        # Joint positions and velocities
        joint_vel_start = 6  # After base DOFs
        self.dof_pos = self._data.qpos[self._joint_ids].copy()
        self.dof_vel = self._data.qvel[joint_vel_start:joint_vel_start + len(self._joint_ids)].copy()
        
        # Projected gravity (simplified approximation)
        self.projected_gravity = np.array([0, 0, -1], dtype=np.float32)  # Should be rotated to body frame
        
        # Commands
        self.commands = self._commands.copy()

    # Reward functions based on legged_gym implementation
    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)."""
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.base_lin_vel[:2]))
        tracking_sigma = getattr(GO2RoughCfg.rewards, 'tracking_sigma', 0.25)
        return np.exp(-lin_vel_error / tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw).""" 
        ang_vel_error = np.square(self.commands[2] - self.base_ang_vel[2])
        tracking_sigma = getattr(GO2RoughCfg.rewards, 'tracking_sigma', 0.25)
        return np.exp(-ang_vel_error / tracking_sigma)
    
    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity."""
        return np.square(self.base_lin_vel[2])
    
    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity."""
        return np.sum(np.square(self.base_ang_vel[:2]))
    
    def _reward_orientation(self):
        """Penalize non flat base orientation."""
        return np.sum(np.square(self.projected_gravity[:2]))
    
    def _reward_base_height(self):
        """Penalize base height away from target."""
        target_height = getattr(GO2RoughCfg.rewards, 'base_height_target', GO2RoughCfg.init_state.pos[2])
        return np.square(self.base_pos[2] - target_height)
    
    def _reward_dof_vel(self):
        """Penalize dof velocities."""
        return np.sum(np.square(self.dof_vel))
    
    def _reward_action_rate(self):
        """Penalize changes in actions."""
        return np.sum(np.square(self._previous_actions))
    
    def _reward_torques(self):
        """Penalize torques (placeholder - actual torques not available in MuJoCo easily)."""
        # Approximate torque penalty based on action magnitude
        return np.sum(np.square(self._previous_actions)) * 0.1
    
    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to limits."""
        # Get joint limits from URDF or set reasonable defaults for GO2
        # These are approximate limits for GO2 joints (hip, thigh, calf for each leg)
        lower_limits = np.array([
            -1.0472, -1.5708, -2.7227,  # FL leg
            -1.0472, -1.5708, -2.7227,  # FR leg  
            -1.0472, -0.5236, -2.7227,  # RL leg
            -1.0472, -0.5236, -2.7227   # RR leg
        ])
        upper_limits = np.array([
            1.0472, 3.4907, -0.83776,   # FL leg
            1.0472, 3.4907, -0.83776,   # FR leg
            1.0472, 4.5379, -0.83776,   # RL leg  
            1.0472, 4.5379, -0.83776    # RR leg
        ])
        
        # Apply soft limits
        soft_limit = getattr(GO2RoughCfg.rewards, 'soft_dof_pos_limit', 0.9)
        mid_range = (lower_limits + upper_limits) / 2
        range_size = upper_limits - lower_limits
        soft_lower = mid_range - 0.5 * range_size * soft_limit
        soft_upper = mid_range + 0.5 * range_size * soft_limit
        
        out_of_limits = np.clip(soft_lower - self.dof_pos, 0, None)  # lower limit violations
        out_of_limits += np.clip(self.dof_pos - soft_upper, 0, None)  # upper limit violations
        return np.sum(out_of_limits)

    # Additional reward functions that might be in the config but not implemented above
    def _reward_tracking_sigma(self):
        """Dummy function - tracking_sigma is a parameter, not a reward."""
        return 0.0

    def _reward_feet_air_time(self):
        """Placeholder for feet air time reward (requires contact detection)."""
        return 0.0
    
    def _reward_collision(self):
        """Placeholder for collision penalty (requires contact detection)."""
        return 0.0
    
    def _reward_feet_stumble(self):
        """Placeholder for stumble penalty (requires contact detection).""" 
        return 0.0
    
    def _reward_stand_still(self):
        """Penalty for standing still when commands are zero."""
        cmd_norm = np.linalg.norm(self.commands[:2])
        if cmd_norm < 0.1:
            return np.linalg.norm(self.base_lin_vel[:2])  # penalize movement when should stand still
        return 0.0
    
    def _reward_dof_acc(self):
        """Penalize dof accelerations (requires previous dof_vel)."""
        # This would need previous step's dof_vel to compute properly
        return 0.0
    
    def _reward_termination(self):
        """Terminal reward/penalty."""
        # Check if robot fell down or went out of bounds
        if self.base_pos[2] < 0.2:  # Robot too low
            return -1.0
        return 0.0


if __name__ == "__main__":
    env = Go2GymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 12))  # 12 actions for GO2
        env.render()
    env.close()
