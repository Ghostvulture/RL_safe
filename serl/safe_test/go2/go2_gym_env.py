from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np
import sys
import os

# Add the path to import go2_rl modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# Add legged_gym to path so it can be imported as a top-level module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'go2_rl'))

#import from legged_gym base - note: go2_rl requires IsaacGym, use simplified config for MuJoCo
print("Note: Using simplified GO2 config (compatible with MuJoCo without IsaacGym dependency)")
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

# Path: /home/xyz/Desktop/xluo/RL_safe/serl/safe_test/go2 -> /home/xyz/Desktop/xluo/RL_safe
_XML_PATH = Path(__file__).parent / "go2_model" / "go2_fixed.xml"

#help function
def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

class Go2GymEnv(MujocoGymEnv):
    def _resample_commands(self):
        # 固定前进速度为 0.5 m/s
        lin_vel_x = 0.5
        lin_vel_y = 0.0
        ang_vel_z = 0.0
        self._commands = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)

    def _tolerance(self, value: float, lower: float, upper: float, sigma: float = None) -> float:
        """简单 tolerance：在 [lower, upper] 内返回 1，否则按距离做指数衰减。"""
        if sigma is None:
            sigma = max((upper - lower) * 0.5, 0.1)
        if lower <= value <= upper:
            return 1.0
        if value < lower:
            dist = lower - value
        else:
            dist = value - upper
        return float(np.exp(- (dist / sigma)))

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: float = 0.25,
        seed: int = 0,
        control_dt: float = 0.02,#TODO:控制频率
        physics_dt: float = 0.002,#0.002
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
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

        # # DEBUG: Print comprehensive MuJoCo model info
        # print(f"DEBUG: GO2 Model loaded successfully!")
        # print(f"DEBUG: Model structure:")
        # print(f"  Bodies: {self._model.nbody}")
        # print(f"  Joints: {self._model.njnt}")
        # print(f"  DOFs (nq): {self._model.nq}")
        # print(f"  Velocities (nv): {self._model.nv}")
        # print(f"  Actuators: {self._model.nu}")
        # print(f"  Lights: {self._model.nlight}")
        # print(f"  Geometries: {self._model.ngeom}")
        
        # # Print all joint names and IDs
        # print(f"DEBUG: All joints in model:")
        # for i in range(self._model.njnt):
        #     joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
        #     joint_type = self._model.jnt_type[i]
        #     qpos_addr = self._model.jnt_qposadr[i]
        #     print(f"  Joint {i}: '{joint_name}' (type: {joint_type}, qpos_addr: {qpos_addr})")
        
        # # Print all actuator names and IDs
        # print(f"DEBUG: All actuators in model:")
        # for i in range(self._model.nu):
        #     actuator_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        #     print(f"  Actuator {i}: '{actuator_name}'")
        
        # # Check if ground plane exists
        # ground_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "ground")  
        # print(f"DEBUG: Ground plane ID: {ground_geom_id}")
        
        # print(f"DEBUG: Render configuration:")
        # print(f"  Mode: {render_mode}")
        # print(f"  Size: {self.render_width}x{self.render_height}")
        
        # # Print our joint mapping
        # print(f"DEBUG: Our joint configuration:")
        # print(f"  Joint names: {self.joint_names}")
        # print(f"  Joint IDs: {self._joint_ids}")
        # print(f"  Actuator names: {self.actuator_names}")
        # print(f"  Actuator IDs: {self._actuator_ctrl_ids}")
        # print(f"  Default angles: {self.default_joint_angles}")
        
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
        
        # Set base orientation (quaternion w, x, y, z) - MuJoCo format
        self._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Standing upright
        
        # Reset joint positions using deploy_mujoco approach (which works correctly)
        # Deploy angles in qpos[7:19] order: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, 
        #                                     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        deploy_angles = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 
                         0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
        
        print(f"DEBUG: Using deploy_mujoco approach for joint angles")
        print(f"  Setting qpos[7:19] with angles: {deploy_angles}")
        
        # Set joint angles directly to qpos[7:19] (matching deploy_mujoco approach)
        for i, angle in enumerate(deploy_angles):
            qpos_idx = 7 + i  # Skip base position (3) and quaternion (4)
            if qpos_idx < len(self._data.qpos):
                self._data.qpos[qpos_idx] = angle
        
        # Reset velocities to zero
        self._data.qvel[:] = 0.0
        
        # Initialize previous actions to zero
        self._previous_actions = np.zeros(self.num_actions, dtype=np.float32)
        
        # 自动采样初始 command（每次 reset 时）
        self._resample_commands()
        
        # Reset time
        self._time = 0.0
        
        # 初始化早期终止标志
        self._was_early_terminated = False
        
        # Forward dynamics to update the state
        mujoco.mj_forward(self._model, self._data)
        
        # DEBUG: Verify the final state
        print(f"DEBUG: Reset complete - Base height: {self._data.qpos[2]:.3f}m (expected: 0.34m)")
        
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
        
        # Convert normalized actions to joint targets using GO2 original approach
        # Deploy default angles: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, 
        #                         RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        deploy_default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
        
        # Action should also be in deploy order to match default angles
        target_joint_pos = self._action_scale * action + deploy_default_angles
        
        # Clip joint targets to REAL GO2 hardware limits from URDF
        # Hip: ±1.0472 rad, Thigh: -1.5708 to 3.4907 rad, Calf: -2.7227 to -0.83776 rad
        joint_limits_lower = np.array([-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, 
                                       -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227])
        joint_limits_upper = np.array([1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 
                                       1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776])
        target_joint_pos = np.clip(target_joint_pos, joint_limits_lower, joint_limits_upper)
        
        # Get current joint positions and velocities for PD control
        current_joint_pos = self._data.qpos[7:]  # Skip base position and quaternion
        current_joint_vel = self._data.qvel[6:]  # Skip base linear and angular velocity
        
        # PD Controller: torques = Kp*(target - current) - Kd*velocity
        Kps = [30.0] *12 
        Kds = [2.0] *12 
        position_error = target_joint_pos - current_joint_pos
        torques = (position_error) * Kps + (np.zeros_like(Kds) - current_joint_vel) * Kds
        
        # Clip torques to hardware limits (from URDF: hip=23.7, thigh=23.7, calf=35.55 Nm)
        torque_limits = np.array([23.7, 23.7, 35.55, 23.7, 23.7, 35.55,
                                  23.7, 23.7, 35.55, 23.7, 23.7, 35.55])
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        # Apply torques to actuators using deploy order mapping
        deploy_actuator_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # Sequential order
        self._data.ctrl[:] = torques
        
        # # DEBUG: Print control inputs before physics step
        print(f"\n=== GO2 CONTROL DEBUG (Step {int(self._time / self.control_dt):4d}) ===")
        
        # Step the physics simulation
        # Use decimation from config (default 4 steps per control step)
        decimation = getattr(GO2RoughCfg.control, 'decimation', 4)
        for _ in range(decimation):
            mujoco.mj_step(self._model, self._data)
        
        # DEBUG: Print joint states after physics step
        self._debug_joint_states()
        
        # Update time
        self._time += self.control_dt

        # 每隔100步采样一次新的command
        if int(self._time / self.control_dt) % 100 == 0:
            self._resample_commands()
        obs = self._compute_observation()
        rew = self._compute_reward()
        
        # 早期终止条件检查 - 加快训练速度
        early_terminated = self._check_early_termination()
        
        # 如果早期终止，设置标志供奖励函数使用
        self._was_early_terminated = early_terminated
        
        # 如果没有早期终止，检查时间限制
        terminated = early_terminated or self.time_limit_exceeded()

        print(f"episode: {int(self._time / self.control_dt):4d} reward: {rew} ")

        return obs, rew, terminated, False, {}

    def _check_early_termination(self):
        """检查早期终止条件，加快训练速度"""
        # 检查是否启用早期终止
        if not getattr(GO2RoughCfg.env, 'enable_early_termination', True):
            return False
            
        # 获取当前机器人状态
        base_pos = self._data.qpos[:3]
        base_quat = self._data.qpos[3:7]
        
        # 1. 高度检查 - 如果机器人掉得太低
        if base_pos[2] < 0.08:  # 低于8cm
            print("！！！！！！！！！！！！！！！！！早期终止: 高度过低")
            return True
        
        # 2. RPY角度检查 - 如果机器人翻倒
        roll, pitch, yaw = self._get_euler_from_quat(base_quat)
        
        # 角度限制 - 给orientation奖励留出发挥空间
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # 约30度
            print(f"！！！！！！！！！！！！！！！！！早期终止: 姿态过倾斜 (roll={roll:.2f}, pitch={pitch:.2f})")
            return True
        
        # 3. 速度检查 - 如果机器人在剧烈旋转
        base_ang_vel = self._data.qvel[3:6]
        if np.linalg.norm(base_ang_vel) > 5.0:  # 角速度过大
            print(f"！！！！！！！！！！！！！！！！！早期终止: 角速度过大 ({np.linalg.norm(base_ang_vel):.2f})")
            return True
        
        # 4. 位置检查 - 如果机器人移动太远
        if abs(base_pos[0]) > 10.0 or abs(base_pos[1]) > 10.0:
            print(f"！！！！！！！！！！！！！！！！！早期终止: 位置超出范围 ({base_pos[0]:.2f}, {base_pos[1]:.2f})")
            return True
            
        # 5. 关节角度检查 - 如果关节超出安全范围
        current_joint_angles = self._data.qpos[7:19]
        joint_limits_lower = np.array([-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, 
                                       -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227])
        joint_limits_upper = np.array([1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 
                                       1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776])
        
        # 检查是否有关节严重超出限制
        violations = 0
        for i in range(12):
            if current_joint_angles[i] < joint_limits_lower[i] - 0.2 or current_joint_angles[i] > joint_limits_upper[i] + 0.2:
                violations += 1
        
        if violations >= 3:  # 如果3个或更多关节超出限制
            print(f"早期终止: {violations}个关节严重超出限制")
            return True
        
        return False

    def set_commands(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """
        Set movement commands for the GO2 robot.
        
        Args:
            lin_vel_x (float): Forward/backward linear velocity (m/s) [positive = forward]
            lin_vel_y (float): Left/right linear velocity (m/s) [positive = left]  
            ang_vel_z (float): Yaw angular velocity (rad/s) [positive = counter-clockwise]
        """
        self._commands = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        print(f"Commands set to: forward={lin_vel_x:.2f} m/s, sideways={lin_vel_y:.2f} m/s, turning={ang_vel_z:.2f} rad/s")

    def render(self):
        # print(f"DEBUG: render() called, viewer exists: {self._viewer is not None}")
        if self._viewer is None:
            print("DEBUG: No viewer available, returning empty frames")
            return []
        
        rendered_frames = []
        for cam_id in self.camera_id:
            try:
                frame = self._viewer.render(render_mode="rgb_array")#, camera_id=cam_id
                rendered_frames.append(frame)
            except Exception as e:
                print(f"ERROR: Failed to render camera {cam_id}: {e}")
        
        return rendered_frames

    # Helper methods.

    def _debug_joint_states(self):
        """Print detailed joint states for debugging."""
        # print("--- Joint States ---")
        
        # Get current joint angles (from qpos[7:19])
        current_joint_angles = self._data.qpos[7:19]  # 12 joints
        
        # Get joint velocities (from qvel[6:18])  
        current_joint_vels = self._data.qvel[6:18]  # 12 joint velocities
        
        # Get applied control commands (from ctrl) - these should be position targets, not torques
        control_commands = self._data.ctrl[:12]  # First 12 actuators in deploy order
        
        # Get actual torques from MuJoCo actuators
        # For position actuators, the torque is stored in actuator_force
        actual_torques = self._data.actuator_force[:12]  # First 12 actuators (deploy order)
        
        joint_names_short = ['FL_hip', 'FL_thigh', 'FL_calf', 'FR_hip', 'FR_thigh', 'FR_calf',
                            'RL_hip', 'RL_thigh', 'RL_calf', 'RR_hip', 'RR_thigh', 'RR_calf']
        
        for i in range(12):
            joint_name = joint_names_short[i]
            angle = current_joint_angles[i]
            vel = current_joint_vels[i]
            torque = actual_torques[i] if i < len(actual_torques) else 0.0
            
        
        # Print base state
        base_pos = self._data.qpos[:3]
        base_quat = self._data.qpos[3:7]
        base_lin_vel = self._data.qvel[:3]
        base_ang_vel = self._data.qvel[3:6]
        roll_dbg, pitch_dbg, _ = self._get_euler_from_quat(base_quat)

        
        print(f"Base pos: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"Base vel: [{base_lin_vel[0]:6.3f}, {base_lin_vel[1]:6.3f}, {base_lin_vel[2]:6.3f}]")
        print(f"Base ang_vel: [{base_ang_vel[0]:6.3f}, {base_ang_vel[1]:6.3f}, {base_ang_vel[2]:6.3f}]")
        print(f"Base RPY: roll={roll_dbg:.3f}, pitch={pitch_dbg:.3f}")

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
        # Properly rotate gravity vector to body frame using quaternion
        projected_gravity = self._quat_rotate_inverse(base_quat, gravity_world)
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
 
        # Get reward scales from config
        self.reward_scales = class_to_dict(GO2RoughCfg.rewards.scales)
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
            # if name == "termination":
            #     continue
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
        
        # Compute each reward component， 直接简单加和
        for i, reward_func in enumerate(self.reward_functions):
            name = self.reward_names[i]
            try:
                rew = reward_func() * self.reward_scales[name]
                print("reward name: {}, reward: {}".format(name, rew))
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
        
        # Store previous dof velocities for acceleration computation
        if not hasattr(self, 'last_dof_vel'):
            self.last_dof_vel = self.dof_vel.copy()
        
        # Store previous actions for action rate computation
        if not hasattr(self, 'last_actions'):
            self.last_actions = self._previous_actions.copy()
        
        # Projected gravity (properly computed with quaternion rotation)
        gravity_world = np.array([0, 0, -1], dtype=np.float32)
        # Properly rotate gravity vector to body frame using quaternion
        self.projected_gravity = self._quat_rotate_inverse(self.base_quat, gravity_world)
        
        # Commands
        self.commands = self._commands.copy()
        
        # Contact forces (placeholder - would need contact detection for MuJoCo)
        # In real implementation, this would come from contact sensors
        self.contact_forces = np.zeros((4, 3), dtype=np.float32)  # 4 feet, 3D forces
        
        # Compute torques from current control commands
        if hasattr(self, '_data') and hasattr(self._data, 'ctrl'):
            self.torques = self._data.ctrl[:12].copy()  # Current control commands as torque proxy
        else:
            self.torques = np.zeros(12, dtype=np.float32)

    # --------------------Reward functions---------------------
    def _reward_alive(self):
        """Reward for staying alive and upright."""
        # Base reward for being alive
        alive_reward = 1.0
        
        # Bonus for maintaining good posture
        if self.base_pos[2] > 0.25:  # Above minimum height
            alive_reward += 0.5
            
        # Bonus for good orientation
        roll, pitch, _ = self._get_euler_from_quat(self.base_quat)
        if abs(roll) < 0.3 and abs(pitch) < 0.3:  # Not too tilted
            alive_reward += 0.5
            
        return alive_reward
    
    def _reward_tracking(self):
        """改进的前进速度奖励：使用 pitch 修正、tolerance、yaw 惩罚、足部接触乘子并放大。"""
        # 当前前向速度（世界/机身 x 轴）
        lin_vel = float(self.base_lin_vel[0])
        # RPY
        _, pitch, _ = self._get_euler_from_quat(self.base_quat)

        # 鼓励身体水平时前进
        forward_aligned = lin_vel * np.cos(pitch)

        # # 使用 tolerance：目标速度范围 [0.5, 1.0] 时给最高奖励
        # target_low = 0.5
        # tol_reward = self._tolerance(forward_aligned, target_low, target_low*2)

        # # 偏航（yaw 角速度）惩罚
        # yaw_rate = float(self.base_ang_vel[2])
        # yaw_penalty = 0.1 * abs(yaw_rate)

        # reward = tol_reward - yaw_penalty
        reward += forward_aligned
        reward = max(reward, 0.0)  # 不让 reward 变为负数（可选策略）
        reward *= 10.0

        return reward
        
    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)."""
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.base_lin_vel[:2]))
        tracking_sigma = getattr(GO2RoughCfg.rewards, 'tracking_sigma', 0.25)
        return np.exp(-lin_vel_error / tracking_sigma)*5
    
    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw).""" 
        ang_vel_error = np.square(self.commands[2] - self.base_ang_vel[2])
        tracking_sigma = getattr(GO2RoughCfg.rewards, 'tracking_sigma', 0.25)
        return np.exp(-ang_vel_error / tracking_sigma)*5
    
    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity."""
        return np.square(self.base_lin_vel[2])
    
    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity."""
        return np.sum(np.square(self.base_ang_vel[:2]))
    
    def _reward_orientation(self):
        """Penalize non flat base orientation."""
        # Use the first two components (x, y) of projected gravity
        # When robot is upright, projected_gravity should be [0, 0, -1]
        # When tilted, x and y components will be non-zero
        orientation_penalty = np.sum(np.square(self.projected_gravity[:2]))
        
        # Debug: Print orientation info occasionally
        if hasattr(self, '_time') and int(self._time / self.control_dt) % 50 == 0:
            roll, pitch, _ = self._get_euler_from_quat(self.base_quat)
            print(f"DEBUG orientation: projected_gravity={self.projected_gravity}, penalty={orientation_penalty:.6f}, roll={roll:.3f}, pitch={pitch:.3f}")
        
        return orientation_penalty
    
    def _reward_base_height(self):
        """Penalize base height away from target."""
        target_height = 0.34  # meters
        return np.square(self.base_pos[2] - target_height)
    
    def _reward_dof_vel(self):
        """Penalize dof velocities."""
        return np.sum(np.square(self.dof_vel))
    
    def _reward_action_rate(self):
        """Penalize changes in actions."""
        if hasattr(self, 'last_actions'):
            action_diff = self._previous_actions - self.last_actions
            self.last_actions = self._previous_actions.copy()
            return np.sum(np.square(action_diff))
        else:
            self.last_actions = self._previous_actions.copy()
            return 0.0
    
    def _reward_torques(self):
        """Penalize high torques (estimated from PD control)."""
        # Estimate torques from PD control law
        if hasattr(self, 'torques'):
            return np.sum(np.square(self.torques))
        else:
            # Fallback: use action magnitude as torque proxy
            return np.sum(np.square(self._previous_actions * 50.0))  # Scale to approximate torque range
    
    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to limits."""
        # GO2 joint limits from URDF (same as used in debug output)
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
        """Reward feet air time based on commanded velocity."""
        # Simulate contact detection based on foot position
        # Get foot positions from forward kinematics (simplified)
        target_air_time = 0.5  # seconds
        
        # Initialize air time tracking if not exists
        if not hasattr(self, 'feet_air_time'):
            self.feet_air_time = np.zeros(4)
            self.last_contacts = np.ones(4, dtype=bool)  # Assume starting on ground
        
        # Simplified contact detection based on base height and leg extension
        # In a real implementation, this would use contact sensors
        contact_threshold = 0.02  # meters above ground
        estimated_foot_height = self.base_pos[2] - 0.3  # Rough estimate
        current_contacts = np.array([estimated_foot_height < contact_threshold] * 4)
        
        # Update air time
        for i in range(4):
            if current_contacts[i]:
                self.feet_air_time[i] = 0.0
            else:
                self.feet_air_time[i] += self.dt
        
        # Reward air time close to target when moving
        rew = 0.0
        if np.linalg.norm(self.commands[:2]) > 0.1:  # Only when commanded to move
            for i in range(4):
                if self.feet_air_time[i] > 0:
                    rew += np.exp(-np.abs(self.feet_air_time[i] - target_air_time))
        
        self.last_contacts = current_contacts
        return rew / 4.0  # Average over all feet
    
    def _reward_collision(self):
        """Penalty for collisions (simplified detection)."""
        # In MuJoCo, we can check if the base is too low or tilted
        collision_penalty = 0.0
        
        # Check if base is too low (collision with ground)
        if self.base_pos[2] < 0.10:  # 降低阈值，给学习更多空间
            collision_penalty += 1.0
        
        # Check if robot is too tilted (potential collision)
        roll, pitch, _ = self._get_euler_from_quat(self.base_quat)
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # Too tilted
            collision_penalty += 0.5
        
        return collision_penalty
    
    def _reward_stumble(self):
        """Penalty for stumbling (rapid foot contacts)."""
        if not hasattr(self, 'last_contacts'):
            return 0.0
        
        # Initialize contact change tracking
        if not hasattr(self, 'contact_changes'):
            self.contact_changes = np.zeros(4)
        
        # Detect rapid contact changes (stumbling)
        current_contacts = self.last_contacts if hasattr(self, 'last_contacts') else np.ones(4, dtype=bool)
        
        stumble_penalty = 0.0
        for i in range(4):
            if hasattr(self, 'last_last_contacts'):
                # Check for rapid on-off-on pattern (stumble indicator)
                if (self.last_last_contacts[i] and 
                    not self.last_contacts[i] and 
                    current_contacts[i]):
                    stumble_penalty += 1.0
        
        # Store contact history
        if hasattr(self, 'last_contacts'):
            self.last_last_contacts = self.last_contacts.copy()
        
        return stumble_penalty
    
    def _reward_feet_contact_forces(self):
        """Reward for appropriate contact forces (placeholder)."""
        # In real implementation, this would use force sensors
        # For now, return small positive reward when on ground
        if self.base_pos[2] > 0.2 and self.base_pos[2] < 0.4:  # Reasonable height
            return 0.1
        return 0.0
    
    def _reward_stand_still(self):
        """Penalty for standing still when commands are zero."""
        cmd_norm = np.linalg.norm(self.commands[:2])
        if cmd_norm < 0.1:
            return np.linalg.norm(self.base_lin_vel[:2])  # penalize movement when should stand still
        return 0.0
    
    def _reward_dof_acc(self):
        """Penalize dof accelerations."""
        if hasattr(self, 'last_dof_vel'):
            dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
            self.last_dof_vel = self.dof_vel.copy()
            return np.sum(np.square(dof_acc))
        else:
            self.last_dof_vel = self.dof_vel.copy()
            return 0.0
    
    def _reward_dof_vel_limits(self):
        """Penalize dof velocities close to limits."""
        # GO2 approximate joint velocity limits (rad/s)
        vel_limits = np.array([
            21.0, 21.0, 33.0,  # FL leg  
            21.0, 21.0, 33.0,  # FR leg
            21.0, 21.0, 33.0,  # RL leg
            21.0, 21.0, 33.0   # RR leg
        ])
        
        soft_limit = 0.9
        vel_violations = np.clip(np.abs(self.dof_vel) - vel_limits * soft_limit, 0, None)
        return np.sum(vel_violations)
    
    def _reward_torque_limits(self):
        """Penalize torques close to limits."""
        # GO2 approximate torque limits (Nm)
        torque_limits = np.array([
            23.7, 23.7, 35.5,  # FL leg
            23.7, 23.7, 35.5,  # FR leg  
            23.7, 23.7, 35.5,  # RL leg
            23.7, 23.7, 35.5   # RR leg
        ])
        
        soft_limit = 0.8
        # Use action magnitude as torque proxy
        torque_violations = np.clip(np.abs(self._previous_actions) * 50.0 - torque_limits * soft_limit, 0, None)
        return np.sum(torque_violations)
    
    def _reward_dof_pos(self):
        """Penalize joint positions away from default."""
        # GO2 default positions (standing pose)
        default_pos = np.array([
            0.0, 0.8, -1.5,   # FL leg
            0.0, 0.8, -1.5,   # FR leg
            0.0, 1.0, -1.5,   # RL leg  
            0.0, 1.0, -1.5    # RR leg
        ])
        
        return np.sum(np.square(self.dof_pos - default_pos))
    
    def _reward_termination(self):
        """Terminal reward/penalty - 只在实际早期终止时给大惩罚"""
        # 检查是否有早期终止标志
        if hasattr(self, '_was_early_terminated') and self._was_early_terminated:
            print("!!!!!!!!!!!!!!!!!!!!!!早期终止！！！给予大惩罚")
            return 5.0  
        else:
            # 正常状态下，给予小的渐进式惩罚以引导行为
            penalty = 0.0
            base_pos = self.base_pos
            roll, pitch, _ = self._get_euler_from_quat(self.base_quat)
            
            # 轻微的渐进式惩罚 - 引导机器人保持良好状态
            if base_pos[2] < 0.2:  # 高度稍低
                penalty += (0.2 - base_pos[2]) * 0.5  # 越低惩罚越大
            
            if abs(roll) > 0.3 or abs(pitch) > 0.3:  # 轻微倾斜
                penalty += (abs(roll) + abs(pitch) - 0.6) * 0.3
            
            # Debug info
            if penalty > 0:
                print(f"DEBUG termination: height={base_pos[2]:.3f}, roll={roll:.3f}, pitch={pitch:.3f}, penalty={penalty:.3f}")
            
            return penalty  # 返回负数作为惩罚
    
    def _get_euler_from_quat(self, quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # quat is [w, x, y, z] in MuJoCo
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def _quat_rotate_inverse(self, quat, vec):
        """Rotate vector by the inverse of a quaternion."""
        # quat is [w, x, y, z] in MuJoCo format
        # Convert to inverse quaternion (conjugate for unit quaternions)
        w, x, y, z = quat
        quat_inv = np.array([w, -x, -y, -z], dtype=np.float32)
        
        # Perform quaternion rotation: q^-1 * vec * q
        # For efficiency, we can use the rotation formula directly
        # v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
        q_xyz = quat_inv[1:4]  # [x, y, z] part
        q_w = quat_inv[0]      # w part
        
        # First cross product: cross(q_xyz, v) + q_w * v
        cross1 = np.cross(q_xyz, vec) + q_w * vec
        
        # Second cross product: cross(q_xyz, cross1)
        cross2 = np.cross(q_xyz, cross1)
        
        # Final result: v + 2 * cross2
        result = vec + 2.0 * cross2
        
        return result.astype(np.float32)


def make_go2_env(render_mode="rgb_array", **kwargs):
    """Factory function to create GO2 environment"""
    return Go2GymEnv(render_mode=render_mode, **kwargs)


if __name__ == "__main__":
    # Demo / smoke-test for the GO2 MuJoCo environment.
    # Improvements over the original:
    # - Use try/finally to ensure env.close() is always called
    # - Capture and print step return values (obs, reward, done)
    # - Use a gentle sinusoidal action pattern instead of full-random actions
    # - Render intermittently to reduce overhead
    import time

    env = Go2GymEnv(render_mode="human")
    obs, _ = env.reset()

    # Set command for forward walking at 0.5 m/s
    env.set_commands(lin_vel_x=0.5, lin_vel_y=0.0, ang_vel_z=0.0)

    steps = 200
    render_every = 2

    try:
        for i in range(steps):
            # Gentle sinusoidal actions to explore movement without destructive randomness
            t = i * env.control_dt
            phases = np.arange(env.num_actions)
            action = 0.2 * np.sin(2.0 * np.pi * 0.5 * t + 0.5 * phases)

            obs, rew, terminated, truncated, info = env.step(action)

            # Print concise debug info
            cmd = obs["state"]["go2/cmd"] if "state" in obs and "go2/cmd" in obs["state"] else env._commands
            print(f"Step {i:04d} reward={rew:.4f} cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] terminated={terminated}")

            # Render less frequently to keep demo responsive
            if (i % render_every) == 0:
                frames = env.render()

            if terminated or truncated:
                print(f"Episode finished at step {i}")
                break

            # Sleep to roughly match real-time control rate (optional)
            time.sleep(max(0.0, env.control_dt * 0.5))
    finally:
        env.close()
