from typing import Optional

from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import SubprocVecEnv

from rl_project.algorithm import ppo2, dqn, a2c, acer, trpo


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
    if algorithm_name == 'TRPO':
        return trpo.get_trpo(
            vec_env=vec_env,
            policy=algorithm_policy,
        )
    return None


def load_model(path: str, algorithm: str):
    from stable_baselines import PPO2, DQN, A2C, ACER, GAIL, TRPO
    if algorithm == 'PPO2':
        return PPO2.load(path)
    if algorithm == 'DQN':
        return DQN.load(path)
    if algorithm == 'A2C':
        return A2C.load(path)
    if algorithm == 'ACER':
        return ACER.load(path)
    if algorithm == 'GAIL':
        return GAIL.load(path)
    if algorithm == 'TRPO':
        return TRPO.load(path)
    return None
