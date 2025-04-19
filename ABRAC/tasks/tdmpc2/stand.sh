# Define TASK
export TASK="g1-stand-v0"

# Define checkpoints to pre-trained low-level policy and obs normalization
export POLICY_PATH="data/stand/torch_model.pt"
export MEAN_PATH="data/stand/mean.npy"
export VAR_PATH="data/stand/var.npy"
export WANDB_ENTITY="1125802866-peking-university"

# Train TD-MPC2
python -m tdmpc2.train disable_wandb=False wandb_entity=${WANDB_ENTITY} exp_name=tdmpc task=humanoid_${TASK} seed=0 policy_path=${POLICY_PATH} mean_path=${MEAN_PATH} var_path=${VAR_PATH}

# # Train DreamerV3
# python -m embodied.agents.dreamerv3.train --configs humanoid_benchmark --run.wandb True --run.wandb_entity [WANDB_ENTITY] --method dreamer --logdir logs --task humanoid_${TASK} --seed 0

# # Train SAC
# python ./jaxrl_m/examples/mujoco/run_mujoco_sac.py --env_name ${TASK} --wandb_entity [WANDB_ENTITY] --seed 0

# # Train PPO (not using MJX)
# python ./ppo/run_sb3_ppo.py --env_name ${TASK} --wandb_entity [WANDB_ENTITY] --seed 0