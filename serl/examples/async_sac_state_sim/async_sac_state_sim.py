#!/usr/bin/env python3

import time
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from flax.training import checkpoints

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)

from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer

# Import GO2 environment
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'safe_test', 'go2'))
from go2_gym_env import Go2GymEnv

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "GO2-v0", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 500, "Maximum length of trajectory.")  # GO2 walking episodes need more steps
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 0, "Sample random actions for this many steps.")  # No random exploration for GO2 walking
flags.DEFINE_integer("training_starts", 1000, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 50, "Number of steps per update the server.")  # Less frequent updates for stability

flags.DEFINE_integer("log_period", 50, "Logging period.")  # Less frequent logging
flags.DEFINE_integer("eval_period", 5000, "Evaluation period.")  # Less frequent evaluation
flags.DEFINE_integer("eval_n_trajs", 3, "Number of trajectories for evaluation.")  # Fewer eval trajectories

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def flatten_observation(obs_dict):
    # Concatenate all observation components
    obs_list = []
    for key in sorted(obs_dict.keys()):  # Sort keys for consistent ordering
        if isinstance(obs_dict[key], dict):
            # Handle nested dict (state)
            for subkey in sorted(obs_dict[key].keys()):
                obs_list.append(obs_dict[key][subkey].flatten())
        else:
            obs_list.append(obs_dict[key].flatten())
    return np.concatenate(obs_list)

class FlattenedGO2Env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate flattened observation space size
        sample_obs = env.observation_space.sample()
        flattened_obs = flatten_observation(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flattened_obs.shape, dtype=np.float32
        )
        
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return flatten_observation(obs_dict), info
        
    def step(self, action):
        obs_dict, reward, done, truncated, info = self.env.step(action)
        return flatten_observation(obs_dict), reward, done, truncated, info

def make_go2_env(render_mode="rgb_array"):
    """Create GO2 gym environment"""
    from mujoco_gym_env import GymRenderingSpec
    
    # Enhanced rendering configuration
    render_spec = GymRenderingSpec(
        width=1024,
        height=768,
        camera_id=0  # Use main camera
    )
    
    env = Go2GymEnv(
        render_mode=render_mode,
        render_spec=render_spec
    )
    env = FlattenedGO2Env(env)
    return env


##############################################################################


def actor(agent: SACAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )
    
    # Initialize WandB logger for actor
    if not FLAGS.debug:
        wandb.init(
            project="go2_sac_training",  # 统一项目名称
            name=f"Actor-{FLAGS.exp_name or 'default'}",  # Actor运行名称
            group="GO2_SAC_Training",
            tags=["SAC", "GO2", "quadruped", "actor", "environment_interaction"],
            config={
                "env": FLAGS.env,
                "algorithm": "SAC",
                "max_steps": FLAGS.max_steps,
                # "random_steps": FLAGS.random_steps,
                "seed": FLAGS.seed,
                "batch_size": FLAGS.batch_size,
                "max_traj_length": FLAGS.max_traj_length,
                "role": "actor",
                "component": "environment_interaction",
            }
        )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    if FLAGS.env == "GO2-v0":
        eval_env = make_go2_env(render_mode="rgb_array")
    else:
        eval_env = gym.make(FLAGS.env)
        if FLAGS.env == "PandaPickCube-v0":
            eval_env = gym.wrappers.FlattenObservation(eval_env)#openai gym wrapper
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False
    
    # Set forward walking command for GO2 training
    if FLAGS.env == "GO2-v0":
        env.set_commands(lin_vel_x=1.0, lin_vel_y=0.0, ang_vel_z=0.0)  # 1 m/s forward
        print("GO2 commands set: Forward walking at 1.0 m/s")

    # training loop
    timer = Timer()
    running_return = 0.0
    episode_length = 0
    episode_count = 0
    total_steps = 0
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    action_magnitudes = []
    
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            # Always use agent policy for GO2 - no random exploration phase
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                deterministic=False,
            )
            actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)

            running_return += reward
            episode_length += 1
            total_steps += 1
            
            # Track action statistics
            action_magnitudes.append(np.linalg.norm(actions))

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done or truncated,
                )
            )

            obs = next_obs
            if done or truncated:
                # Log episode statistics
                episode_rewards.append(running_return)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # WandB logging per episode
                if not FLAGS.debug:
                    wandb.log({
                        "actor/episode_reward": running_return,
                        # "actor/episode_length": episode_length,
                        "actor/episode_count": episode_count,
                        # "actor/total_steps": total_steps,
                        "actor/avg_action_magnitude": np.mean(action_magnitudes[-episode_length:]) if action_magnitudes else 0.0,
                        # "actor/exploration_phase": step < FLAGS.random_steps,
                    }, step=total_steps)
                
                running_return = 0.0
                episode_length = 0
                obs, _ = env.reset()
                
                # Reset forward walking command after each episode
                if FLAGS.env == "GO2-v0":
                    env.set_commands(lin_vel_x=1.0, lin_vel_y=0.0, ang_vel_z=0.0)  # 1 m/s forward

        if FLAGS.render:
            env.render()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)
            
            # Additional WandB logging for training progress
            if not FLAGS.debug and episode_rewards:
                recent_episodes = min(10, len(episode_rewards))
                wandb.log({
                    "actor/avg_episode_reward_10": np.mean(episode_rewards[-recent_episodes:]),
                    "actor/max_episode_reward": np.max(episode_rewards),
                    "actor/min_episode_reward": np.min(episode_rewards),
                    "actor/avg_episode_length_10": np.mean(episode_lengths[-recent_episodes:]) if episode_lengths else 0,
                    # "actor/steps_per_second": FLAGS.log_period / timer.get_average_times().get("total", 1.0),
                    # "actor/replay_buffer_size": len(data_store) if hasattr(data_store, '__len__') else 0,
                }, step=total_steps)


##############################################################################


def learner(rng, agent: SACAgent, replay_buffer, replay_iterator):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="go2_sac_training",  # 统一项目名称
        description=f"Learner-{FLAGS.exp_name or 'default'}",  # Learner运行名称
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # Train the networks
        with timer.context("sample_replay_buffer"):
            batch = next(replay_iterator)

        with timer.context("train"):
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)#FLAGS.utd_ratio
            agent = jax.block_until_ready(agent)

            # publish the updated network
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            # Log SAC training metrics
            log_dict = {}
            log_dict.update(update_info)
            
            # Add timer information
            log_dict.update({"timer": timer.get_average_times()})
            
            # Add learning progress metrics
            log_dict["learner/update_steps"] = update_steps
            log_dict["learner/replay_buffer_size"] = len(replay_buffer)
            
            # Log SAC-specific metrics if available
            if "actor_loss" in update_info:
                log_dict["sac/actor_loss"] = update_info["actor_loss"]
            if "critic_loss" in update_info:
                log_dict["sac/critic_loss"] = update_info["critic_loss"]
            # if "temp_loss" in update_info:
            #     log_dict["sac/temperature_loss"] = update_info["temp_loss"]
            # if "temperature" in update_info:
            #     log_dict["sac/temperature"] = update_info["temperature"]
            if "entropy" in update_info:
                log_dict["sac/entropy"] = update_info["entropy"]
            
            wandb_logger.log(log_dict, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.env == "GO2-v0":
        if FLAGS.render:
            env = make_go2_env(render_mode="human")
        else:
            env = make_go2_env(render_mode="rgb_array")
    else:
        if FLAGS.render:
            env = gym.make(FLAGS.env, render_mode="human")
        else:
            env = gym.make(FLAGS.env)

        if FLAGS.env == "PandaPickCube-v0":
            env = gym.wrappers.FlattenObservation(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="replay_buffer",
            preload_rlds_path=FLAGS.preload_rlds_path,
        )
        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio,
            },
            device=sharding.replicate(),
        )
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator=replay_iterator,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
