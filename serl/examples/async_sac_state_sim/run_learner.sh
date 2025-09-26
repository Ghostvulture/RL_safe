export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --env GO2-v0 \
    --exp_name=go2_sac_walking_training \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --batch_size 128 \
    --max_traj_length 500 \
    --debug # wandb is disabled when debug
