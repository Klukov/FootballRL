import re


def _get_run_name(
        algorithm: str,
        scenario_number: int,
        number_of_steps: int,
        number_of_envs: int,
) -> str:
    return "{algorithm}_scenario-{scenario}_{steps}M-steps_{envs}-env".format(
        algorithm=algorithm,
        scenario=scenario_number,
        steps=str(number_of_steps / 1e6).replace(".", "") if number_of_steps < 1e6
        else re.sub("\..*", "", str(number_of_steps / 1e6)),
        envs=number_of_envs,
    )
