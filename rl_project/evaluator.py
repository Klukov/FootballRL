from absl import logging as logger
from stable_baselines.common import BaseRLModel

from rl_project.environment import create_demo_env, SCENARIO_MAP


def evaluate_model(
        model: BaseRLModel,
        scenario_number: int,
        accuracy: int,
        reward: str,
        representation: str,
        stacked: bool,
        render: bool,
) -> int:
    """
    Function returns average goal difference after number of scenario runs - accuracy parameter
    """

    if model is None:
        raise ValueError("Trained model must be given. Please run script with --help option")
    logger.info("EVALUATOR STARTED")
    env = create_demo_env(
        level=SCENARIO_MAP.get(scenario_number, 'academy_empty_goal_close'),
        reward_experiment=reward,
        representation=representation,
        stacked=stacked,
        render=render,
    )
    number_of_runs = int(0)
    total_reward = float(0)
    while number_of_runs < accuracy:
        obs = env.reset()
        index = 0
        done = False
        rewards = float(0.0)
        while not done:
            index = index + 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        logger.info("EVALUATOR: run-{run} ended with reward: {reward}".format(
            run=number_of_runs,
            reward=rewards,
        ))
        total_reward = total_reward + rewards
        number_of_runs = number_of_runs + 1
    return total_reward
