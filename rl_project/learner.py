from stable_baselines.common import BaseRLModel


def create_rl_algorithm_model(
        algorithm: str,
        algorithm_policy: str,
        scenario_number: int,
        number_of_envs: int,
        representation: str,
        stacked: bool,
) -> BaseRLModel:

    from rl_project import environment
    from rl_project.algorithm import get_configured_rl_model, configure_tensorflow
    from rl_project.environment import SCENARIO_MAP

    env = environment.create_training_env(
        number_of_processes=number_of_envs,
        level=SCENARIO_MAP.get(scenario_number, 'academy_empty_goal_close'),
        representation=representation,
        stacked=stacked,
    )
    configure_tensorflow()
    return get_configured_rl_model(
        vec_env=env,
        algorithm_name=algorithm,
        algorithm_policy=algorithm_policy,
    )
