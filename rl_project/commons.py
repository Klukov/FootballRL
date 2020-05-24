import os

from stable_baselines import logger
from stable_baselines.bench import monitor
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from gfootball.env import create_environment


def create_training_env(
    number_of_processes,
    level='academy_empty_goal_close',
    state='extracted_stacked',
    reward_experiment='scoring',
    dump_scores=False,
    dump_full_episodes=False,
    render=False,
):
  """
  Meaning of all variables you can find in footbal.gfootball.examples.run_ppo2.py
  :return: stable_baselines.common.vec_env.subproc_vec_env.SubprocVecEnv
  """
  return SubprocVecEnv([
    (lambda _i=i: create_single_football_env(
      _i,
      level,
      state,
      reward_experiment,
      dump_scores,
      dump_full_episodes,
      render
    ))
    for i in range(number_of_processes)
  ])


def create_single_football_env(
    process_number=0,
    level='academy_empty_goal_close',
    state='extracted_stacked',
    reward_experiment='scoring',
    dump_scores=False,
    dump_full_episodes=False,
    render=False,
):
  """
  Creates gfootball environment.
  Meaning of all variables you can find in footbal.gfootball.examples.run_ppo2.py
  """
  env = create_environment(
    env_name=level, stacked=('stacked' in state),
    rewards=reward_experiment,
    logdir=logger.get_dir(),
    write_goal_dumps=dump_scores and (process_number == 0),
    write_full_episode_dumps=dump_full_episodes and (process_number == 0),
    render=render and (process_number == 0),
    dump_frequency=50 if render and process_number == 0 else 0)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(process_number)))
  return env
