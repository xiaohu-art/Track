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

class root_deviation(Termination[MotionLibG1]):
    def __init__(self, env, max_distance: float = 0.4) -> None:
        super().__init__(env)
        self.max_distance = torch.tensor(max_distance, device=self.device)
        self.robot = self.command_manager.robot

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_root_translation = self.command_manager.root_pos_w[timestep]
        ref_root_translation.add_(self.command_manager.env_origin)
        root_pos_w = self.robot.data.root_pos_w
        deviation = (root_pos_w - ref_root_translation).norm(dim=1, keepdim=True)
        return deviation > self.max_distance
    
class root_rot_deviation(Termination[MotionLibG1]):
    def __init__(self, env, threshold: float = 0.8) -> None:
        super().__init__(env)
        self.threshold = threshold
        self.robot = self.command_manager.robot

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_quat_w = self.command_manager.root_quat_w[timestep]
        root_quat_w = self.robot.data.root_quat_w

        ref_projected_gravity_b = quat_apply_inverse(ref_quat_w, self.robot.data.GRAVITY_VEC_W)
        projected_gravity_b = quat_apply_inverse(root_quat_w, self.robot.data.GRAVITY_VEC_W)

        diff = (projected_gravity_b[:, 2] - ref_projected_gravity_b[:, 2]).abs().unsqueeze(-1)
        return diff > self.threshold
    
class track_kp_error(Termination[MotionLibG1]):
    def __init__(self, env, threshold: float = 0.4, body_names: str = ".*") -> None:
        super().__init__(env)
        self.threshold = threshold
        self.robot = self.command_manager.robot
        self.body_indices = self.robot.find_bodies(body_names)[0]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        timestep = self.command_manager.episode_start_frames + self.env.episode_length_buf - 1
        ref_keypoints = self.command_manager.body_pos_w[timestep][:, self.body_indices]
        ref_keypoints.add_(self.command_manager.env_origin[:, None])

        body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]

        # diff = (ref_keypoints - body_pos_global).norm(dim=-1)    # (num_envs, num_bodies)
        # return diff.mean(-1, True) > self.threshold
        diff = (ref_keypoints[:, :, 2] - body_pos_global[:, :, 2]).abs()
        return (diff > self.threshold).any(dim=-1, keepdim=True)