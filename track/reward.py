import os
import joblib
import torch
from pathlib import Path

from active_adaptation.envs.mdp.base import Reward
from isaaclab.utils.math import (
    quat_apply_inverse, quat_inv, quat_mul, 
    subtract_frame_transforms, quat_error_magnitude
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

from track.command import MotionLibG1

ADAPTIVE_SIGMA = {
    "sigma": {
        "tracking_anchor_pos": 0.16,
        "tracking_anchor_quat": 0.16,
        "tracking_qpos": 0.16,
        "tracking_kp_pos": 0.36,
        "tracking_kp_quat": 0.36,
        "tracking_kp_lin_vel": 1.0,
        "tracking_kp_ang_vel": 3.14,
    },
    "params": {
        "alpha": 1e-3
    }
}

def _bind_adaptive_sigma(env):
    if hasattr(env, "_adaptive_sigma"):
        return

    env._adaptive_sigma = {k: torch.tensor(v, device=env.device) for k, v in ADAPTIVE_SIGMA["sigma"].items()}
    env._error_ema = {k: torch.tensor(v, device=env.device) for k, v in env._adaptive_sigma.items()}
    env._alpha = ADAPTIVE_SIGMA["params"]["alpha"]

    def _update_adaptive_sigma_method(self, error, term):
        self._error_ema[term] = self._error_ema[term] * (1 - self._alpha) + error * self._alpha
        self._adaptive_sigma[term] = torch.minimum(self._adaptive_sigma[term], self._error_ema[term])
    
    import types
    env._update_adaptive_sigma = types.MethodType(_update_adaptive_sigma_method, env)

class tracking_anchor_pos(Reward[MotionLibG1]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        _bind_adaptive_sigma(env)
        self.robot = self.command_manager.robot
        self.anchor_body_index = self.command_manager.anchor_body_index

    def compute(self) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_anchor_pos_w = self.command_manager.body_pos_w[timestep][:, self.anchor_body_index]
        ref_anchor_pos_w.add_(self.command_manager.env_origin[:, None])
        anchor_pos_w = self.robot.data.body_link_pos_w[:, self.anchor_body_index]
        error = (anchor_pos_w - ref_anchor_pos_w).square().sum(-1)
        # reward = torch.exp(- error / self.sigma)
        reward = torch.exp(- error / self.env._adaptive_sigma["tracking_anchor_pos"])
        self.env._update_adaptive_sigma(error.mean(), "tracking_anchor_pos")
        return reward
    
class tracking_anchor_quat(Reward[MotionLibG1]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        _bind_adaptive_sigma(env)
        self.robot = self.command_manager.robot
        self.anchor_body_index = self.command_manager.anchor_body_index

    def compute(self) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_anchor_quat_w = self.command_manager.body_quat_w[timestep][:, self.anchor_body_index]
        anchor_quat_w = self.robot.data.body_link_quat_w[:, self.anchor_body_index]
        error = (quat_error_magnitude(anchor_quat_w, ref_anchor_quat_w) ** 2)
        # reward = torch.exp(- error / self.sigma)
        reward = torch.exp(- error / self.env._adaptive_sigma["tracking_anchor_quat"])
        self.env._update_adaptive_sigma(error.mean(), "tracking_anchor_quat")
        return reward
    
class tracking_qpos(Reward[MotionLibG1]):
    def __init__(self, env, weight: float = 1.0, joint_names: str = ".*") -> None:
        super().__init__(env, weight)
        _bind_adaptive_sigma(env)
        self.robot = self.command_manager.robot
        self.joint_indices, self.joint_names = self.robot.find_joints(joint_names, preserve_order=True)

    def compute(self) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_qpos = self.command_manager.joint_pos[timestep][:, self.joint_indices]
        qpos = self.robot.data.joint_pos[:, self.joint_indices]
        error = (qpos - ref_qpos).square().mean(-1, True)
        # reward = torch.exp(- error / self.sigma)
        reward = torch.exp(- error / self.env._adaptive_sigma["tracking_qpos"])
        self.env._update_adaptive_sigma(error.mean(), "tracking_qpos")
        return reward
    
class tracking_kp_pos(Reward[MotionLibG1]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        _bind_adaptive_sigma(env)
        self.robot = self.command_manager.robot
        self.keypoint_body_index = self.command_manager.keypoint_body_index

    def compute(self) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_keypoints = self.command_manager.body_pos_w[timestep][:, self.keypoint_body_index]
        ref_keypoints.add_(self.command_manager.env_origin[:, None])

        body_pos_global = self.robot.data.body_link_pos_w[:, self.keypoint_body_index]

        error = (ref_keypoints - body_pos_global).square().sum(-1).mean(-1, True)
        # reward = torch.exp(- error / self.sigma)
        reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_pos"])
        self.env._update_adaptive_sigma(error.mean(), "tracking_kp_pos")
        return reward
    
class tracking_kp_quat(Reward[MotionLibG1]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        _bind_adaptive_sigma(env)
        self.robot = self.command_manager.robot
        self.keypoint_body_index = self.command_manager.keypoint_body_index

    def compute(self) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_keypoints = self.command_manager.body_quat_w[timestep][:, self.keypoint_body_index]

        body_quat_w = self.robot.data.body_link_quat_w[:, self.keypoint_body_index]
        error = (quat_error_magnitude(body_quat_w, ref_keypoints) ** 2).mean(-1, True)

        reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_quat"])
        self.env._update_adaptive_sigma(error.mean(), "tracking_kp_quat")
        return reward

class tracking_kp_lin_vel(Reward[MotionLibG1]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        _bind_adaptive_sigma(env)
        self.robot = self.command_manager.robot
        self.keypoint_body_index = self.command_manager.keypoint_body_index

    def compute(self) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_lin_vel = self.command_manager.body_lin_vel_w[timestep][:, self.keypoint_body_index]
        body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.keypoint_body_index]
        error = (body_lin_vel_w - ref_lin_vel).square().sum(-1).mean(-1, True)
        reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_lin_vel"])
        self.env._update_adaptive_sigma(error.mean(), "tracking_kp_lin_vel")
        return reward

class tracking_kp_ang_vel(Reward[MotionLibG1]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        _bind_adaptive_sigma(env)
        self.robot = self.command_manager.robot
        self.keypoint_body_index = self.command_manager.keypoint_body_index

    def compute(self) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_ang_vel = self.command_manager.body_ang_vel_w[timestep][:, self.keypoint_body_index]
        body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.keypoint_body_index]
        error = (body_ang_vel_w - ref_ang_vel).square().sum(-1).mean(-1, True)
        reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_ang_vel"])
        self.env._update_adaptive_sigma(error.mean(), "tracking_kp_ang_vel")
        return reward

class feet_slip(Reward[MotionLibG1]):
    def __init__(
        self, env, body_names: str, weight: float = 1.0
    ):
        super().__init__(env, weight)
        self.robot = self.command_manager.robot
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.robot.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

    def compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.02
        )
        feet_vel = self.robot.data.body_lin_vel_w[:, self.articulation_body_ids, :2]
        slip = (in_contact * feet_vel.norm(dim=-1).square()).sum(dim=1, keepdim=True)
        return -slip