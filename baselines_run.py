from rl_project import ppo2
from rl_project import environment

import logging
import re
import time

if __name__ == '__main__':
    number_of_envs = int(1)
    total_number_of_steps = int(1e5)

    run_name = "ppo_test_model_{steps}M_steps_{envs}-env".format(
        steps=str(total_number_of_steps / 1e6).replace(".", "") if total_number_of_steps < 1e6
                else re.sub("\..*", "", str(total_number_of_steps / 1e6)),
        envs=number_of_envs,
    )
    logging.basicConfig(filename=run_name + ".log", level=logging.DEBUG)
    logger = logging.getLogger(run_name)
    logger.info("STARTED")

    time_start = time.time()
    env = environment.create_training_env(number_of_envs)
    trained_model = ppo2.learn_ppo2(vec_env=env, total_number_of_steps=total_number_of_steps)
    time_elapsed = time.time() - time_start
    logger.info("Time elapsed " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))

    trained_model.save(save_path=run_name)
