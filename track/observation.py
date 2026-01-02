import os
import joblib
import torch
from pathlib import Path

from active_adaptation.envs.mdp.base import Observation
from isaaclab.utils.math import (
    quat_apply, quat_apply_inverse, 
    quat_inv, quat_mul, matrix_from_quat,
    subtract_frame_transforms, quat_error_magnitude
)
from typing_extensions import TYPE_CHECKING

from track.command import MotionLibG1

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std
    
class root_quat_w(Observation[MotionLibG1]):
    def __init__(self, env, noise_std):
        super().__init__(env)
        self.noise_std = noise_std

    def compute(self) -> torch.Tensor:
        self.root_quat_w = self.command_manager.robot.data.root_quat_w
        root_quat_w = random_noise(self.root_quat_w, self.noise_std)
        return root_quat_w.reshape(self.num_envs, -1)

class ref_root_quat(Observation[MotionLibG1]):
    def __init__(self, env):
        super().__init__(env)
        self.robot = self.command_manager.robot

    def compute(self):  
        current_frames = self.command_manager.episode_start_frames + self.env.episode_length_buf
        current_frames = torch.min(current_frames, self.command_manager.episode_end_frames - 1)
        ref_root_quat_w = self.command_manager.root_quat_w[current_frames]
        root_quat_w = self.robot.data.root_quat_w
        
        quat_error = quat_mul(quat_inv(root_quat_w), ref_root_quat_w)
        return quat_error.reshape(self.num_envs, -1)

class ref_qpos(Observation[MotionLibG1]):
    def __init__(self, env, joint_names=".*"):
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.joint_indices, self.joint_names = self.robot.find_joints(joint_names, preserve_order=True)

    def compute(self) -> torch.Tensor:
        current_frames = self.command_manager.episode_start_frames + self.env.episode_length_buf
        current_frames = torch.min(current_frames, self.command_manager.episode_end_frames - 1)
        ref_qpos = self.command_manager.joint_pos[current_frames]
        ref_qpos = ref_qpos[:, self.joint_indices]
        return ref_qpos.reshape(self.num_envs, -1)

class ref_kp_pos_gap(Observation[MotionLibG1]):
    def __init__(self, env):
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.keypoint_body_index = self.env.command_manager.keypoint_body_index

        self.ref_kp_pos = self.env.command_manager.body_pos_w[:, self.keypoint_body_index]    # (num_frames, num_keypoints, 3)
        self.ref_kp_quat = self.env.command_manager.body_quat_w[:, self.keypoint_body_index]

    def compute(self):
        current_frames = self.command_manager.episode_start_frames + self.env.episode_length_buf
        current_frames = torch.min(current_frames, self.command_manager.episode_end_frames - 1)
        ref_kp_pos = self.ref_kp_pos[current_frames]       # (num_envs, num_keypoints, 3)
        ref_kp_pos.add_(self.command_manager.env_origin[:, None])
        ref_kp_quat = self.ref_kp_quat[current_frames]

        body_kp_pos = self.robot.data.body_link_pos_w[:, self.keypoint_body_index]
        body_kp_quat = self.robot.data.body_link_quat_w[:, self.keypoint_body_index]

        pos, _ = subtract_frame_transforms(body_kp_pos, body_kp_quat, ref_kp_pos, ref_kp_quat)
        return pos.reshape(self.num_envs, -1)

class ref_kp_quat(Observation[MotionLibG1]):
    def __init__(self, env):
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.keypoint_body_index = self.command_manager.keypoint_body_index

        self.ref_kp_pos = self.command_manager.body_pos_w[:, self.keypoint_body_index]    # (num_frames, num_keypoints, 3)
        self.ref_kp_quat = self.command_manager.body_quat_w[:, self.keypoint_body_index]

    def compute(self):
        current_frames = self.command_manager.episode_start_frames + self.env.episode_length_buf
        current_frames = torch.min(current_frames, self.command_manager.episode_end_frames - 1)
        ref_kp_pos = self.ref_kp_pos[current_frames]       # (num_envs, num_keypoints, 3)
        ref_kp_pos.add_(self.command_manager.env_origin[:, None])
        ref_kp_quat = self.ref_kp_quat[current_frames]

        body_kp_pos = self.robot.data.body_link_pos_w[:, self.keypoint_body_index]
        body_kp_quat = self.robot.data.body_link_quat_w[:, self.keypoint_body_index]

        _, quat = subtract_frame_transforms(body_kp_pos, body_kp_quat, ref_kp_pos, ref_kp_quat)
        return quat.reshape(self.num_envs, -1)