import os
import joblib
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple

from active_adaptation.envs.mdp.base import Command
import active_adaptation as aa
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    yaw_quat,
    quat_inv,
    quat_apply,
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
        self.episode_motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._init_adaptive()

    def _init_adaptive(self):
        self.bin_size = 50
        self.adaptive_kernel_size = 1
        self.adaptive_lambda = 0.8
        self.adaptive_uniform_ratio = 0.3

        self.bin_count_per_motion = self.motion_length // self.bin_size + 1
        self.max_bin_count = int(self.bin_count_per_motion.max().item())
        self.bin_failed_count = torch.zeros(
            (self.num_motions, self.max_bin_count),
            dtype=torch.long,
            device=self.device
        )
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
    
    def adaptive_sampling(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive sampling based on failure statistics.
        
        Args:
            env_ids: Environment IDs to sample for.
            
        Returns:
            motion_ids: Sampled motion IDs.
            sampled_bins: Sampled bin IDs.
        """
        # Record failures from terminated environments
        episode_lengths = self.env.stats["episode_len"]
        termination = self.env._compute_termination()

        mask = termination[env_ids]
        if mask.any():
            terminated_envs = env_ids[mask.nonzero(as_tuple=True)[0]]
            lengths = episode_lengths[terminated_envs].squeeze(-1)
            
            motion_ids = self.episode_motion_ids[terminated_envs]

            start_frames = self.start_frames[motion_ids]
            current_frames = self.episode_start_frames[terminated_envs] + lengths

            bin_ids = ((current_frames - start_frames) // self.bin_size).long()
            max_bins = self.bin_count_per_motion[motion_ids] - 1
            bin_ids = torch.minimum(bin_ids, max_bins)
            flat_index = motion_ids * self.max_bin_count + bin_ids
            update_cnt = torch.bincount(
                flat_index, minlength=self.num_motions * self.max_bin_count
            ).view(self.num_motions, self.max_bin_count)
            self.bin_failed_count += update_cnt
        
        # Sample motions based on failure statistics
        motion_scores = self.bin_failed_count.sum(dim=1)
        motion_scores = motion_scores / (self.bin_count_per_motion.float() + 1e-6)

        total_fail = motion_scores.sum()
        if total_fail < 1e-6:
            motion_ids = torch.randint(0, self.num_motions, (env_ids.shape[0],), device=self.device, dtype=torch.long)
        else:
            motion_probs = (1 - self.adaptive_uniform_ratio) * (motion_scores / total_fail) + \
                           (self.adaptive_uniform_ratio / self.num_motions)
            motion_ids = torch.multinomial(motion_probs, env_ids.shape[0], replacement=True)

        # Sample bins based on failure statistics with smoothing
        bin_scores = self.bin_failed_count[motion_ids, :self.max_bin_count].float()
        bin_probs = bin_scores + self.adaptive_uniform_ratio / self.bin_count_per_motion[motion_ids, None]
        bin_probs = torch.nn.functional.pad(
            bin_probs.unsqueeze(1),
            (0, self.adaptive_kernel_size - 1),
            mode="replicate",
        )
        bin_probs = torch.nn.functional.conv1d(bin_probs, self.kernel.view(1, 1, -1)).squeeze(1)
        
        bin_indices = torch.arange(self.max_bin_count, device=self.device).unsqueeze(0).expand(motion_ids.shape[0], -1)
        bin_mask = bin_indices < self.bin_count_per_motion[motion_ids, None]

        bin_probs = torch.where(bin_mask, bin_probs, torch.zeros_like(bin_probs))
        bin_probs = bin_probs / bin_probs.sum(dim=1, keepdim=True)
        sampled_bins = torch.multinomial(bin_probs, num_samples=1).squeeze(1)
        
        return motion_ids, sampled_bins
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample initial states for given environments.
        
        Args:
            env_ids: Environment IDs to initialize.
            
        Returns:
            Dictionary containing initial root states for robots.
        """
        # Adaptive sampling
        motion_ids, sampled_bins = self.adaptive_sampling(env_ids)

        # Compute start and end frames
        start_frames = self.start_frames[motion_ids]
        end_frames = self.end_frames[motion_ids]
        if self.env.training:
            start_frames += sampled_bins * self.bin_size

        self.episode_motion_ids[env_ids] = motion_ids
        self.episode_start_frames[env_ids] = start_frames
        self.episode_end_frames[env_ids] = end_frames
        
        init_root_pos_w = self.root_pos_w[start_frames].to(self.device) + self.env_origin[env_ids]
        if self.env.training:
            rand_pos_samples = torch.zeros((env_ids.shape[0], 3), device=self.device)
            rand_pos_samples[:, 0].uniform_(self.pose_range["x"][0], self.pose_range["x"][1])
            rand_pos_samples[:, 1].uniform_(self.pose_range["y"][0], self.pose_range["y"][1])
            rand_pos_samples[:, 2].uniform_(self.pose_range["z"][0], self.pose_range["z"][1])
            init_root_pos_w = init_root_pos_w + rand_pos_samples

        init_root_quat_w = self.root_quat_w[start_frames].to(self.device)
        if self.env.training:
            rand_quat_samples = torch.zeros((env_ids.shape[0], 3), device=self.device)
            rand_quat_samples[:, 0].uniform_(self.pose_range["roll"][0], self.pose_range["roll"][1])
            rand_quat_samples[:, 1].uniform_(self.pose_range["pitch"][0], self.pose_range["pitch"][1])
            rand_quat_samples[:, 2].uniform_(self.pose_range["yaw"][0], self.pose_range["yaw"][1])
            orientation_delta = quat_from_euler_xyz(rand_quat_samples[:, 0], rand_quat_samples[:, 1], rand_quat_samples[:, 2])
            init_root_quat_w = quat_mul(orientation_delta, init_root_quat_w)

        init_root_state = self.init_root_state[env_ids]     # (num_envs, 3 + 4 + 6) root position, root orientation, root linear velocity and root angular velocity
        init_root_state[:, :3] = init_root_pos_w
        init_root_state[:, 3:7] = init_root_quat_w

        if aa.get_backend() == "isaac":
            init_root_state[:, 7:10] = self.root_lin_vel_w[start_frames]
            init_root_state[:, 10:] = self.root_ang_vel_w[start_frames]

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