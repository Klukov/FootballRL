import time

from stable_baselines.common import BaseRLModel

from stablebaselines_based import _get_run_name


def create_rl_algorithm_model(
        algorithm: str,
        algorithm_policy: str,
        scenario_number: int,
        number_of_envs: int = 8,
        representation: str = 'extracted',
        stacked: bool = True,
) -> BaseRLModel:
    from stablebaselines_based import environment
    from stablebaselines_based.algorithm import _get_configured_rl_model, _configure_tensorflow
    from stablebaselines_based.environment import SCENARIO_MAP

    env = environment.create_training_env(
        number_of_processes=number_of_envs,
        level=SCENARIO_MAP.get(scenario_number, 'academy_empty_goal_close'),
        representation=representation,
        stacked=stacked,
    )
    _configure_tensorflow()
    return _get_configured_rl_model(
        vec_env=env,
        algorithm_name=algorithm,
        algorithm_policy=algorithm_policy,
    )


def run_training(
        algorithm: str = 'PPO2',
        algorithm_policy='CnnPolicy',
        scenario_number: int = 17,
        number_of_steps: int = int(1e6),
        number_of_envs: int = 8,
) -> None:
    """
    FUNCTION FOR SIMPLE TRAINING BY NOT USING BASH SCRIPTS
    """

    import logging

    run_name = _get_run_name(
        algorithm=algorithm,
        scenario_number=scenario_number,
        number_of_steps=number_of_steps,
        number_of_envs=number_of_envs,
    )
    logging.basicConfig(filename=run_name + ".log", level=logging.DEBUG)
    logger = logging.getLogger(run_name)
    logger.info("STARTED: " + run_name)
    time_start = time.time()

    trained_model = create_rl_algorithm_model(
        algorithm=algorithm,
        algorithm_policy=algorithm_policy,
        scenario_number=scenario_number,
        number_of_envs=number_of_envs,
    ).learn(number_of_steps)

    time_elapsed = time.time() - time_start
    logger.info("Time elapsed " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    trained_model.save(save_path=run_name)
