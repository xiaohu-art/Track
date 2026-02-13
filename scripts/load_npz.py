"""This script processes multiple motions from a pkl file and saves collected simulation data back to the original pkl file.

.. code-block:: bash

    # Usage
    python load_pkl.py --npz_dir xxx --output_file xxx --input_fps 50 --output_fps 50 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
import glob
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Process multiple motions from npz file and save collected data back to pkl.")
parser.add_argument("--npz_dir", type=str, required=True, help="The path to the directory containing npz files.")
parser.add_argument("--output_file", type=str, required=True, help="The path to the output pkl file.")
parser.add_argument("--input_fps", type=int, default=50, help="The fps of the input motion.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import joblib
from tqdm import tqdm

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
import active_adaptation as aa
aa.set_backend("isaac")

from active_adaptation.assets.humanoid import G1_WAIST_UNLOCKED_CFG

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot = G1_WAIST_UNLOCKED_CFG.isaaclab().replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_data: dict,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        motion_key: str,
    ):
        self.motion_data = motion_data
        self.motion_key = motion_key
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the motion data dict."""

        motion = torch.from_numpy(self.motion_data["qpos"])
        motion = motion.to(torch.float32).to(self.device)

        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_key}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def process_single_motion(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str], 
                         motion_data: dict, motion_key: str) -> dict:
    """Processes a single motion and returns collected simulation data."""
    # Load motion
    motion = MotionLoader(
        motion_data=motion_data,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        motion_key=motion_key,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
            
            print(f"[INFO]: Motion {motion_key} processed successfully")
            break
    
    return log


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop for multiple motions."""
    # Load the pkl file containing multiple motions
    print(f"[INFO]: Loading motions from {args_cli.npz_dir}")
    pattern = os.path.join(args_cli.npz_dir, "**", "*.npz")
    files = sorted(glob.glob(pattern, recursive=True))

    print(f"[INFO]: Processing {len(files)} motions")

    # Process each motion
    processed_motions = {}
    for file in tqdm(files, desc="Processing motions"):
        print(f"[INFO]: Processing motion: {file}")
        motion_data = np.load(file, allow_pickle=True)
        rel = os.path.relpath(file, args_cli.npz_dir).replace("\\", "/")
        parts = rel.split("/")
        motion_key = f"{parts[0]}_{parts[1]}"
        
        # Process the motion and collect simulation data
        simulation_data = process_single_motion(sim, scene, joint_names, motion_data, motion_key)
        
        # Replace original motion data with collected simulation data
        processed_motions[motion_key] = simulation_data
    
    # Save the updated pkl file
    print(f"[INFO]: Saving updated motions to {args_cli.output_file}")
    joblib.dump(processed_motions, args_cli.output_file)
    print(f"[INFO]: Successfully processed and saved {len(processed_motions)} motions")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()      # type: ignore
    # close sim app
    simulation_app.close()
