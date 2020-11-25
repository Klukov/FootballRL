import logging

from rl_project.evaluator import run_evaluation
from rl_project.trainer import run_training

if __name__ == '__main__':
    ALGORITHM = 'PPO2'
    SCENARIOS = (17, 14, 13,)
    TIME_STEPS = (10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000,)
    ACCURACY = 1000
    NUMBER_OF_ENVS = 8
    for scenario in SCENARIOS:
        for number_of_time_steps in TIME_STEPS:
            log = logging.getLogger()
            for log_handler in log.handlers[:]:  # remove all old handlers
                log.removeHandler(log_handler)
            run_training(
                algorithm=ALGORITHM,
                scenario_number=scenario,
                number_of_steps=number_of_time_steps,
                number_of_envs=NUMBER_OF_ENVS,
            )
            # log = logging.getLogger()
            # for log_handler in log.handlers[:]:  # remove all old handlers
            #     log.removeHandler(log_handler)
            run_evaluation(
                algorithm=ALGORITHM,
                scenario_number=scenario,
                number_of_steps=number_of_time_steps,
                number_of_envs=NUMBER_OF_ENVS,
                accuracy=ACCURACY,
            )
