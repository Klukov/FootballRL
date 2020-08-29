import os

from gfootball.env import create_environment
from stable_baselines import logger
from stable_baselines.bench import monitor
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def create_demo_env(
        level='academy_empty_goal_close',
        reward_experiment='scoring,checkpoints',
        stacked=True,
        representation='extracted',
):
    return SubprocVecEnv([
        (lambda _i=i: _create_single_football_env(
            process_number=_i,
            level=level,
            stacked=stacked,
            representation=representation,
            reward_experiment=reward_experiment,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            write_video=False,
            dump_frequency=1,
            render=True,
        ))
        for i in range(1)
    ])


def create_training_env(
        number_of_processes,
        level='academy_empty_goal_close',
        stacked=True,
        representation='extracted',
        reward_experiment='scoring,checkpoints',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        write_video=False,
        dump_frequency=1,
) -> SubprocVecEnv:
    """
  Meaning of all variables you can find in footbal.gfootball.examples.run_ppo2.py
  :return: stable_baselines.common.vec_env.subproc_vec_env.SubprocVecEnv
  """
    return SubprocVecEnv([
        (lambda _i=i: _create_single_football_env(
            process_number=_i,
            level=level,
            stacked=stacked,
            representation=representation,
            reward_experiment=reward_experiment,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            write_video=write_video,
            dump_frequency=dump_frequency,
            render=True,
        ))
        for i in range(number_of_processes)
    ])


def _create_single_football_env(
        level,
        stacked,
        representation,
        reward_experiment,
        write_goal_dumps,
        write_full_episode_dumps,
        write_video,
        dump_frequency,
        render,
        process_number=0,
):
    """
  Creates gfootball environment.
  Meaning of all variables you can find in footbal.gfootball.examples.run_ppo2.py
  """
    env = create_environment(
        env_name=level,
        stacked=stacked,
        representation=representation,
        rewards=reward_experiment,
        logdir=logger.get_dir(),
        write_goal_dumps=write_goal_dumps and (process_number == 0),
        write_full_episode_dumps=write_full_episode_dumps and (process_number == 0),
        write_video=write_video,
        render=render and (process_number == 0),
        dump_frequency=dump_frequency if render and process_number == 0 else 0)
    env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(process_number)))
    return env
