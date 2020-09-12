import os

from bidict import bidict
from gfootball.env import create_environment
from stable_baselines import logger
from stable_baselines.bench import monitor
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


SCENARIO_MAP = bidict({
    1: '11_vs_11_competition',
    2: '11_vs_11_stochastic',
    3: 'academy_corner',
    4: 'academy_empty_goal',
    5: 'academy_run_to_score_with_keeper',
    6: '11_vs_11_easy_stochastic',
    7: '1_vs_1_easy',
    8: 'academy_counterattack_easy',
    9: 'academy_pass_and_shoot_with_keeper',
    10: 'academy_single_goal_versus_lazy',
    11: '11_vs_11_hard_stochastic',
    12: '5_vs_5',
    13: 'academy_counterattack_hard',
    14: 'academy_run_pass_and_shoot_with_keeper',
    15: '11_vs_11_kaggle',
    16: 'academy_3_vs_1_with_keeper',
    17: 'academy_empty_goal_close',
    18: 'academy_run_to_score',
})


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
            render=False,
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
