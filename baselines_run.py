import re
import time
from typing import Optional

from absl import app
from absl import flags
from absl import logging as logger
from stable_baselines.common import ActorCriticRLModel
from stable_baselines.common.vec_env import SubprocVecEnv

from rl_project import environment
from rl_project.algorithm import ppo2
from rl_project.environment import SCENARIO_MAP

FLAGS = flags.FLAGS

flags.DEFINE_integer('scenario_number', int(17),
                     'Defines scenario number - look at Readme.md', int(1), int(18))
flags.DEFINE_enum('algorithm', 'ppo2', ['ppo2'],
                  'Algorithm to be used for training - only some algorithms from stable-baselines')
flags.DEFINE_integer('number_of_envs', int(8),
                     'Number of env run parallelly to get result', int(1))
flags.DEFINE_integer('number_of_steps', int(1e5),
                     'Total number of steps of the algorithm', int(1))


def main(_):
    def get_run_name() -> str:
        return "{algorithm}_scenario-{scenario}_{steps}M-steps_{envs}-env".format(
            algorithm=FLAGS.algorithm,
            scenario=FLAGS.scenario_number,
            steps=str(FLAGS.number_of_steps / 1e6).replace(".", "") if FLAGS.number_of_steps < 1e6
            else re.sub("\..*", "", str(FLAGS.number_of_steps / 1e6)),
            envs=FLAGS.number_of_envs,
        )

    def train(
            vec_env: SubprocVecEnv,
            total_number_of_steps: int,
            algorithm_name: str,
    ) -> Optional[ActorCriticRLModel]:
        if algorithm_name == 'ppo2':
            return ppo2.learn_ppo2(vec_env=vec_env, total_number_of_steps=total_number_of_steps)
        return None

    run_name = get_run_name()
    logger.get_absl_handler().use_absl_log_file(run_name, './')
    logger.info("STARTED")
    time_start = time.time()
    env = environment.create_training_env(
        number_of_processes=FLAGS.number_of_envs,
        level=SCENARIO_MAP.get(FLAGS.scenario_number, 'academy_empty_goal_close'),
        representation=FLAGS.representation,
        stacked=FLAGS.stacked,
    )
    trained_model = train(vec_env=env, total_number_of_steps=FLAGS.number_of_steps, algorithm_name=FLAGS.algorithm)
    time_elapsed = time.time() - time_start
    logger.info("Time elapsed " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    trained_model.save(save_path=run_name)


if __name__ == '__main__':
    app.run(main)
