from typing import Optional

from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import SubprocVecEnv

from rl_project.algorithm import ppo2, dqn, a2c, acer, gail, trpo


def train(
        vec_env: SubprocVecEnv,
        total_number_of_steps: int,
        algorithm_name: str,
        algorithm_policy: str,
) -> Optional[BaseRLModel]:
    if algorithm_name == 'PPO2':
        return ppo2.learn_ppo2(
            vec_env=vec_env,
            policy=algorithm_policy,
            total_number_of_steps=total_number_of_steps,
        )
    if algorithm_name == 'DQN':
        return dqn.learn_dqn(
            env=vec_env,
            policy=algorithm_policy,
            total_number_of_steps=total_number_of_steps,
        )
    if algorithm_name == 'A2C':
        return a2c.learn_a2c(
            vec_env=vec_env,
            policy=algorithm_policy,
            total_number_of_steps=total_number_of_steps,
        )
    if algorithm_name == 'ACER':
        return acer.learn_acer(
            vec_env=vec_env,
            policy=algorithm_policy,
            total_number_of_steps=total_number_of_steps,
        )
    if algorithm_name == 'GAIL':
        return gail.learn_gail(
            vec_env=vec_env,
            policy=algorithm_policy,
            total_number_of_steps=total_number_of_steps,
        )
    if algorithm_name == 'TRPO':
        return trpo.learn_trpo(
            vec_env=vec_env,
            policy=algorithm_policy,
            total_number_of_steps=total_number_of_steps,
        )
    return None
