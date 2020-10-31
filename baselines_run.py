import re
import time
from typing import Optional

from absl import app
from absl import flags
from absl import logging as logger
from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import SubprocVecEnv

from rl_project import environment
from rl_project.algorithm import ppo2, td3, dqn
from rl_project.environment import SCENARIO_MAP

FLAGS = flags.FLAGS

flags.DEFINE_integer('scenario_number', int(17),
                     'Defines scenario number - look at Readme.md', int(1), int(19))
flags.DEFINE_enum('algorithm', 'ppo2', ['ppo2', 'td3', 'dqn'],
                  'Algorithm to be used for training - only some algorithms from stable-baselines')
flags.DEFINE_string('algorithm_policy', 'CnnPolicy',
                    'The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy...')
flags.DEFINE_integer('number_of_envs', int(1),
                     'Number of env run parallelly to get result. Available only for ', int(1))
flags.DEFINE_integer('number_of_steps', int(1e5),
                     'Total number of steps of the algorithm', int(1))
flags.DEFINE_enum('representation', 'extracted', ['extracted', 'pixels', 'pixels_gray', 'simple115v2'],
                  'Definition of the representation used to build the observation. More info in gfootball.env.__init__')
flags.DEFINE_boolean('stacked', True, 'If True, stack 4 observations, otherwise, only the last observation is returned '
                                      'by the environment. Stacking is only possible when representation is one of the '
                                      'following: "pixels", "pixels_gray" or "extracted"')


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
            algorithm_policy: str,
    ) -> Optional[BaseRLModel]:
        if algorithm_name == 'ppo2':
            return ppo2.learn_ppo2(
                vec_env=vec_env,
                policy=algorithm_policy,
                total_number_of_steps=total_number_of_steps,
            )
        if algorithm_name == 'td3':
            return td3.learn_td3(
                env=vec_env,
                policy=algorithm_policy,
                total_number_of_steps=total_number_of_steps,
            )
        if algorithm_name == 'dqn':
            return dqn.learn_dqn(
                env=vec_env,
                policy=algorithm_policy,
                total_number_of_steps=total_number_of_steps,
            )
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
    trained_model = train(
        vec_env=env,
        total_number_of_steps=FLAGS.number_of_steps,
        algorithm_name=FLAGS.algorithm,
        algorithm_policy=FLAGS.algorithm_policy,
    )
    time_elapsed = time.time() - time_start
    logger.info("Time elapsed " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    trained_model.save(save_path=run_name)


if __name__ == '__main__':
    app.run(main)
