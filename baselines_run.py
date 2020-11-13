import re
import time

from absl import app
from absl import flags
from absl import logging as logger


FLAGS = flags.FLAGS

flags.DEFINE_integer('scenario_number', int(17),
                     'Defines scenario number - look at Readme.md', int(1), int(19))
flags.DEFINE_enum('algorithm', 'PPO2', ['PPO2', 'DQN', 'A2C', 'ACER', 'GAIL', 'TRPO'],
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

    run_name = get_run_name()
    logger.get_absl_handler().use_absl_log_file(run_name, './')
    logger.info("STARTED")
    time_start = time.time()

    from rl_project.learner import create_rl_algorithm_model
    trained_model = create_rl_algorithm_model(
        algorithm=FLAGS.algorithm,
        algorithm_policy=FLAGS.algorithm_policy,
        scenario_number=FLAGS.scenario_number,
        number_of_envs=FLAGS.number_of_envs,
        representation=FLAGS.representation,
        stacked=FLAGS.stacked,
    ).learn(FLAGS.number_of_steps)
    time_elapsed = time.time() - time_start
    logger.info("Time elapsed " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    trained_model.save(save_path=run_name)


if __name__ == '__main__':
    app.run(main)
