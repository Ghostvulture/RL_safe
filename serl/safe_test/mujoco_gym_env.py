from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        hfield: Optional[np.ndarray] = None,
        light_params: Optional[dict] = None,
        texture_path: Optional[str] = None,
        friction: Optional[tuple] = None,
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        # 应用hfield（高度场）数据
        if hfield is not None:
            if hasattr(self._model, "hfield_nrow") and self._model.hfield_nrow > 0:
                nrow = self._model.hfield_nrow
                ncol = self._model.hfield_ncol
                assert hfield.shape == (nrow, ncol), f"hfield shape must be ({nrow}, {ncol})"
                self._model.hfield_data[:] = hfield.ravel()

        # 应用自定义光照参数
        if light_params is not None:
            # 只修改第一个光源
            if self._model.nlight > 0:
                lid = 0
                if "direction" in light_params:
                    self._model.light_dir[lid] = light_params["direction"]
                if "intensity" in light_params:
                    self._model.light_diffuse[lid] = light_params["intensity"]
                if "color" in light_params:
                    self._model.light_ambient[lid] = light_params["color"]

        # 应用自定义地面贴图
        if texture_path is not None:
            # 只修改第一个texture（假设为地面）
            if self._model.ntex > 0:
                tid = 0
                self._model.tex_rgb[tid] = mujoco.load_texture(texture_path)

        # 应用摩擦力参数
        if friction is not None:
            # 只修改第一个geom（假设为地面）
            if self._model.ngeom > 0:
                gid = 0
                self._model.geom_friction[gid] = friction

        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def time_limit_exceeded(self) -> bool:
        return self._data.time >= self._time_limit

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random
