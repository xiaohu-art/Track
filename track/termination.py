import os
import joblib
import torch
from pathlib import Path

from active_adaptation.envs.mdp.base import Termination
from isaaclab.utils.math import quat_apply_inverse

from track.command import MotionLibG1

class success(Termination[MotionLibG1]):
    def __init__(self, env) -> None:
        super().__init__(env)

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        current_frames = self.command_manager.episode_start_frames + self.env.episode_length_buf
        return (current_frames >= self.command_manager.episode_end_frames).unsqueeze(1)

class anchor_xy_deviation(Termination[MotionLibG1]):
    def __init__(self, env, max_distance: float = 0.5) -> None:
        super().__init__(env)
        self.max_distance = torch.tensor(max_distance, device=self.device)
        self.robot = self.command_manager.robot
        self.anchor_body_index = self.command_manager.anchor_body_index

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_anchor_pos_w = self.command_manager.body_pos_w[timestep][:, self.anchor_body_index]
        ref_anchor_pos_w.add_(self.command_manager.env_origin[:, None])

        anchor_pos_w = self.robot.data.body_link_pos_w[:, self.anchor_body_index]
        deviation = torch.norm(anchor_pos_w[..., :2] - ref_anchor_pos_w[..., :2], dim=-1)
        return deviation > self.max_distance

class anchor_z_deviation(Termination[MotionLibG1]):
    def __init__(self, env, max_distance: float = 0.4) -> None:
        super().__init__(env)
        self.max_distance = torch.tensor(max_distance, device=self.device)
        self.robot = self.command_manager.robot
        self.anchor_body_index = self.command_manager.anchor_body_index

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_anchor_pos_w = self.command_manager.body_pos_w[timestep][:, self.anchor_body_index]
        ref_anchor_pos_w.add_(self.command_manager.env_origin[:, None])

        anchor_pos_w = self.robot.data.body_link_pos_w[:, self.anchor_body_index]
        deviation = (anchor_pos_w[:, :, 2] - ref_anchor_pos_w[:, :, 2]).abs()
        return deviation > self.max_distance

class anchor_rot_deviation(Termination[MotionLibG1]):
    def __init__(self, env, threshold: float = 0.8) -> None:
        super().__init__(env)
        self.threshold = threshold
        self.robot = self.command_manager.robot
        self.anchor_body_index = self.command_manager.anchor_body_index

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_anchor_quat_w = self.command_manager.body_quat_w[timestep][:, self.anchor_body_index]

        anchor_quat_w = self.robot.data.body_link_quat_w[:, self.anchor_body_index]

        gravity_vec_w = torch.tensor([0., 0., -1.], device=self.device).repeat(self.num_envs, 1, 1)

        ref_projected_gravity_b = quat_apply_inverse(ref_anchor_quat_w, gravity_vec_w)
        projected_gravity_b = quat_apply_inverse(anchor_quat_w, gravity_vec_w)

        diff = (projected_gravity_b[:, :, 2] - ref_projected_gravity_b[:, :, 2]).abs()
        return diff > self.threshold
    
class track_kp_z_error(Termination[MotionLibG1]):
    def __init__(self, env, threshold: float = 0.4, body_names: str = ".*") -> None:
        super().__init__(env)
        self.threshold = threshold
        self.robot = self.command_manager.robot
        self.body_indices = self.robot.find_bodies(body_names)[0]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        aligned_body_pos_w, _ = self.command_manager.get_aligned_body_state(timestep)
        ref_keypoints = aligned_body_pos_w[:, self.body_indices]

        body_pos_global = self.robot.data.body_link_pos_w[:, self.body_indices]

        diff = (ref_keypoints[:, :, 2] - body_pos_global[:, :, 2]).abs()
        return (diff > self.threshold).any(dim=-1, keepdim=True)