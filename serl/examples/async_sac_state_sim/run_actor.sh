export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_sac_state_sim.py "$@" \
    --actor \
    --render \
    --env GO2-v0 \
    --exp_name=go2_sac_walking_training \
    --seed 0 \
    --random_steps 1000 \
    --max_traj_length 500 \
    # --debug
