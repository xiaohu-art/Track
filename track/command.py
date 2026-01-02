import os
import joblib
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple

from active_adaptation.envs.mdp.base import Command
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

DATA_ROOT = Path(__file__).parents[1] / "data"

class MotionLib(Command):
    def __init__(
            self, 
            env,
            dataset: Union[List[str], str],
            occlusion: str,
            pose_range: Dict[str, Tuple[float, float]],
            joint_range: Tuple[float, float],
            anchor_body: Optional[str] = None,
            keypoint_body: Optional[List[str]] = None,
        ):
        super().__init__(env)
        self.robot = env.scene["robot"]
        self.env_origin = self.env.scene.env_origins

        self.pose_range = pose_range
        self.joint_range = joint_range

        self.anchor_body_index = self.robot.find_bodies(anchor_body)[0]
        self.keypoint_body_index = self.robot.find_bodies(keypoint_body)[0]

        occlusion_path = DATA_ROOT / occlusion
        occlusion_keys = list(joblib.load(occlusion_path).keys())

        # Support both single string (backward compatibility) and list of strings
        if isinstance(dataset, str):
            dataset = [dataset]
        
        # Load and merge all datasets
        data = {}
        for dataset_name in dataset:
            motion_clip = DATA_ROOT / f"{dataset_name}.pkl"
            dataset_data = joblib.load(motion_clip)
            dataset_data = {k.replace("_stageii", "_poses"): v for k, v in dataset_data.items()}
            dataset_data = {k: v for k, v in dataset_data.items() if k not in occlusion_keys}

            # first_key = list(dataset_data.keys())[0]
            # print(first_key)
            # dataset_data = {first_key: dataset_data[first_key]}
            
            data.update(dataset_data)
        
        print(f"Loaded motion clips from {len(dataset)} dataset(s)")

        self.load_data(data)
        assert len(self.robot.body_names) == self.body_pos_w.shape[1]
        assert len(self.robot.joint_names) == self.joint_pos.shape[1]

        self.num_frames = int(self.joint_pos.shape[0])
        print(f"Loaded {len(data)} motion clips with {self.num_frames} frames.")

        self.episode_start_frames = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_end_frames = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _choose_start_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
        start_frames = self.start_frames[motion_ids]

        motion_length = self.motion_length[motion_ids]
        bin_size = 50
        max_bins = ((motion_length - 1) // bin_size).clamp_min(0)
        r = torch.rand_like(max_bins, dtype=torch.float32)
        bin_ids = torch.floor(r * (max_bins.to(torch.float32) + 1.0)).to(torch.long)
        start_frames += bin_ids * bin_size
        return start_frames
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.randint(0, self.num_motions, (env_ids.shape[0],), device=self.device, dtype=torch.long)
        
        start_frames = self._choose_start_frames(motion_ids)
        end_frames = self.end_frames[motion_ids]

        self.episode_start_frames[env_ids] = start_frames
        self.episode_end_frames[env_ids] = end_frames
        
        rand_pos_samples = sample_uniform(self.pose_range["x"][0], self.pose_range["x"][1], (env_ids.shape[0], 3), device=self.device)
        init_root_pos_w = self.root_pos_w[start_frames].to(self.device) + self.env_origin[env_ids]
        init_root_pos_w += rand_pos_samples

        rand_quat_samples = sample_uniform(self.pose_range["roll"][0], self.pose_range["roll"][1], (env_ids.shape[0], 3), device=self.device)
        orientation_delta = quat_from_euler_xyz(rand_quat_samples[:, 0], rand_quat_samples[:, 1], rand_quat_samples[:, 2])
        init_root_quat_w = self.root_quat_w[start_frames].to(self.device)
        init_root_quat_w = quat_mul(orientation_delta, init_root_quat_w)

        init_root_state = self.init_root_state[env_ids]     # (num_envs, 3 + 4 + 6) root position, root orientation, root linear velocity and root angular velocity
        init_root_state[:, :3] = init_root_pos_w
        init_root_state[:, 3:7] = init_root_quat_w

        random_joint_samples = sample_uniform(self.joint_range[0], self.joint_range[1], (env_ids.shape[0], self.robot.num_joints), device=self.device)
        init_joint_pos = self.joint_pos[start_frames].to(self.device)
        init_joint_pos += random_joint_samples

        joint_pos = init_joint_pos
        joint_vel = self.joint_vel[start_frames].to(self.device)
        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids = slice(None),
            env_ids=env_ids
        )
        
        return {"robot": init_root_state}
        
    def load_data(self, data):
        self.motion_length = []
        self.joint_pos = []
        self.joint_vel = []
        self.body_pos_w = []
        self.body_quat_w = []
        self.body_lin_vel_w = []
        self.body_ang_vel_w = []

        pbar = tqdm(data.items())
        for k, motion in pbar:
            pbar.set_description(f"Loading {k}: ")
            joint_pos = torch.from_numpy(motion["joint_pos"])
            joint_vel = torch.from_numpy(motion["joint_vel"])
            body_pos_w = torch.from_numpy(motion["body_pos_w"])
            body_quat_w = torch.from_numpy(motion["body_quat_w"])
            body_lin_vel_w = torch.from_numpy(motion["body_lin_vel_w"])
            body_ang_vel_w = torch.from_numpy(motion["body_ang_vel_w"])

            self.motion_length.append(joint_pos.shape[0])
            self.joint_pos.append(joint_pos)
            self.joint_vel.append(joint_vel)
            self.body_pos_w.append(body_pos_w)
            self.body_quat_w.append(body_quat_w)
            self.body_lin_vel_w.append(body_lin_vel_w)
            self.body_ang_vel_w.append(body_ang_vel_w)

        self.motion_length = torch.tensor(self.motion_length)
        self.joint_pos = torch.cat(self.joint_pos, dim=0).float().to(self.device)
        self.joint_vel = torch.cat(self.joint_vel, dim=0).float().to(self.device)
        self.body_pos_w = torch.cat(self.body_pos_w, dim=0).float().to(self.device)
        self.body_quat_w = torch.cat(self.body_quat_w, dim=0).float().to(self.device)
        self.body_lin_vel_w = torch.cat(self.body_lin_vel_w, dim=0).float().to(self.device)
        self.body_ang_vel_w = torch.cat(self.body_ang_vel_w, dim=0).float().to(self.device)

        self.root_pos_w = self.body_pos_w[:, 0]
        self.root_quat_w = self.body_quat_w[:, 0]
        self.root_lin_vel_w = self.body_lin_vel_w[:, 0]
        self.root_ang_vel_w = self.body_ang_vel_w[:, 0]

        self.num_motions = len(data)
        self.num_frames = self.joint_pos.shape[0]

        self.start_frames = torch.cat([torch.zeros(1), self.motion_length.cumsum(dim=0)[:-1]]).long().to(self.device)
        self.end_frames = self.motion_length.cumsum(dim=0).long().to(self.device)
        self.motion_length = self.motion_length.to(self.device)

    # def update(self):
    #     current_frames = self.episode_start_frames + self.env.episode_length_buf
    #     current_frames = torch.min(current_frames, self.episode_end_frames - 1)
        
    #     root_state = self.robot.data.root_state_w.clone()
    #     root_state[:, :3] = self.root_pos_w[current_frames] + self.env_origin
    #     root_state[:, 3:7] = self.root_quat_w[current_frames]
    #     root_state[:, 7:10] = self.root_lin_vel_w[current_frames]
    #     root_state[:, 10:] = self.root_ang_vel_w[current_frames]
    #     self.robot.write_root_state_to_sim(root_state)

    #     joint_pos = self.joint_pos[current_frames]
    #     joint_vel = self.joint_vel[current_frames]
    #     self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

class MotionLibG1(MotionLib):
    
    def __init__(
            self, 
            env,
            dataset: List[str],
            occlusion: str,
            pose_range: Dict[str, Tuple[float, float]],
            joint_range: Tuple[float, float],
            anchor_body: str = "torso_link",
            keypoint_body: List[str] = [
                                        "pelvis",
                                        "left_hip_pitch_link", "right_hip_pitch_link", 
                                        "left_knee_link", "right_knee_link", 
                                        "left_ankle_roll_link", "right_ankle_roll_link", 
                                        "left_shoulder_roll_link", "right_shoulder_roll_link", 
                                        "left_elbow_link", "right_elbow_link", 
                                        "left_wrist_yaw_link", "right_wrist_yaw_link"
                                        ],
        ):
        super().__init__(
            env,
            dataset,
            occlusion,
            pose_range,
            joint_range,
            anchor_body,
            keypoint_body,
        )