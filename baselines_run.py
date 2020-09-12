from rl_project.algorithm import ppo2
from rl_project import environment

import logging
import re
import time

from rl_project.environment import SCENARIO_MAP

if __name__ == '__main__':
    scenario_number = int(17)
    algorithm = 'ppo2'
    number_of_envs = int(8)
    total_number_of_steps = int(1e5)

    run_name = "{algorithm}_scenario-{scenario}_{steps}M-steps_{envs}-env".format(
        algorithm=algorithm,
        scenario=scenario_number,
        steps=str(total_number_of_steps / 1e6).replace(".", "") if total_number_of_steps < 1e6
                else re.sub("\..*", "", str(total_number_of_steps / 1e6)),
        envs=number_of_envs,
    )
    logging.basicConfig(filename=run_name + ".log", level=logging.DEBUG)
    logger = logging.getLogger(run_name)
    logger.info("STARTED")

    time_start = time.time()
    env = environment.create_training_env(
        number_of_processes=number_of_envs,
        level=SCENARIO_MAP.get(scenario_number, 'academy_empty_goal_close'),
    )
    trained_model = ppo2.learn_ppo2(vec_env=env, total_number_of_steps=total_number_of_steps)
    time_elapsed = time.time() - time_start
    logger.info("Time elapsed " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))

    trained_model.save(save_path=run_name)
