from stable_baselines.common import BaseRLModel

from rl_project import _get_run_name
from rl_project.environment import create_demo_env, SCENARIO_MAP


def evaluate_model(
        model: BaseRLModel,
        scenario_number: int,
        accuracy: int,
        logging,
        reward: str = 'scoring',
        representation: str = 'extracted',
        stacked: bool = True,
        render: bool = False,
) -> int:
    """
    Function returns average goal difference after number of scenario runs - accuracy parameter
    """

    if model is None:
        raise ValueError("Trained model must be given")
    logging("EVALUATOR STARTED")
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
        logging("EVALUATOR: run-{run} ended with reward: {reward}".format(
            run=number_of_runs,
            reward=rewards,
        ))
        total_reward = total_reward + rewards
        number_of_runs = number_of_runs + 1
    return total_reward


def run_evaluation(
        algorithm: str = 'PPO2',
        scenario_number: int = 17,
        number_of_steps: int = int(1e6),
        number_of_envs: int = 8,
        accuracy: int = 1000,
):
    import logging

    run_name = _get_run_name(
        algorithm=algorithm,
        scenario_number=scenario_number,
        number_of_steps=number_of_steps,
        number_of_envs=number_of_envs,
    )

    run_name_with_evaluation_mark = "EVALUATOR-" + run_name
    logging.basicConfig(filename=run_name_with_evaluation_mark + ".log", level=logging.DEBUG)
    logger = logging.getLogger(run_name_with_evaluation_mark)
    logger.info("STARTED: " + run_name_with_evaluation_mark)

    from rl_project.algorithm import load_model
    total_reward = evaluate_model(
        model=load_model(run_name, algorithm=algorithm),
        scenario_number=scenario_number,
        accuracy=accuracy,
        logging=logger.info,
    )

    logger.info(
        "EVALUATOR: ended with {runs} runs and receive in total reward: {reward}. Average reward: {average}".format(
            runs=accuracy,
            reward=total_reward,
            average=total_reward / accuracy,
        ))
