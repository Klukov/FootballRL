from typing import Optional

from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import SubprocVecEnv

from rl_project.algorithm import ppo2, dqn, a2c, acer, gail, trpo


def get_configured_rl_model(
        vec_env: SubprocVecEnv,
        algorithm_name: str,
        algorithm_policy: str,
) -> Optional[BaseRLModel]:
    if algorithm_name == 'PPO2':
        return ppo2.get_ppo2(
            vec_env=vec_env,
            policy=algorithm_policy,
        )
    if algorithm_name == 'DQN':
        return dqn.get_dqn(
            env=vec_env,
            policy=algorithm_policy,
        )
    if algorithm_name == 'A2C':
        return a2c.get_a2c(
            vec_env=vec_env,
            policy=algorithm_policy,
        )
    if algorithm_name == 'ACER':
        return acer.get_acer(
            vec_env=vec_env,
            policy=algorithm_policy,
        )
    if algorithm_name == 'GAIL':
        return gail.get_gail(
            vec_env=vec_env,
            policy=algorithm_policy,
        )
    if algorithm_name == 'TRPO':
        return trpo.get_trpo(
            vec_env=vec_env,
            policy=algorithm_policy,
        )
    return None


def configure_tensorflow() -> None:
    """
    Config taken from gfootball repo, check gfootball.examples.run_ppo2.py
    """
    import tensorflow.compat.v1 as tf
    import multiprocessing
    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
