# Installation

1. Install [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) by downloading the latest release and unzip it to a desired location `$ISAACSIM_PATH`.
2. Clone [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and setup a conda environment:
    ```bash
    git clone git@github.com:isaac-sim/IsaacLab.git # SSH recommended
    conda create -n lab python=3.11
    conda activate lab
    cd IsaacLab
    ln -s $ISAACSIM_PATH _isaac_sim
    ./isaaclab.sh -c lab
    ./isaaclab.sh -i none # install without additional RL libraries

    # reactivate the environment
    conda deactivate
    conda activate lab
    echo $PYTHONPATH
    ```
    You should see the isaac-sim related dependencies are added to `$PYTHONPATH`.
3. `pip install -U torch torchvision tensordict torchrl==0.10`
4. Clone [active_adaptation](https://github.com/Agent-3154/active-adaptation.git) and checkout to branch `v0.4.2`
    ```bash
    git clone git@github.com:Agent-3154/active-adaptation.git
    cd active-adaptation
    git checkout v0.4.2
    pip install -e .
    ```
5. Clone this repository
    ```bash
    git clone https://github.com/xiaohu-art/Track.git
    cd Track
    pip install -e .
    ```

# Training & Evaluation
1. Acquire training reference motion
2. Train the reference motion tracking policy:
    ```bash
    cd active-adaptation/scripts
    python train_ppo.py \
        task=motion \
        algo=ppo_track \
        task.num_envs=<num envs> \
        task.name=<task name> \             # task name for wandb
        total_frames=<total frames> \       # Total number of frames to collect
        checkpoint_path=<checkpoint path>   # resume training from checkpoint
    ```
3. Play the trained policy:
    ```bash
    cd active-adaptation/scripts
    python play.py \
        task=motion \
        algo=ppo_track \
        task.num_envs=<num envs> \
        checkpoint_path=<checkpoint path> \
        export_policy=<true or false>   # whether to export the policy to onnx
    ```