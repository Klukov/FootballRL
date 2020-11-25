from absl import app
from absl import flags
from absl import logging as logger

FLAGS = flags.FLAGS

flags.DEFINE_enum('algorithm', 'PPO2', ['PPO2', 'DQN', 'A2C', 'ACER', 'GAIL', 'TRPO'],
                  'Algorithm used for model training - only some algorithms from stable-baselines')
flags.DEFINE_string('path', None, 'path to stored model')
flags.DEFINE_integer('scenario_number', int(17),
                     'Defines scenario number (look at Readme.md), which will be used for evaluation', int(1), int(19))
flags.DEFINE_integer('accuracy', int(1e3),
                     'Number of runs to evaluate model correctness', int(1))
flags.DEFINE_boolean('render', False,
                     'if enabled then game screen pop up and you can observe how agent behaves')
flags.DEFINE_enum('reward', 'scoring', ['scoring', 'checkpoints', 'scoring,checkpoints'],
                  'Option defines how each run should be rewarded')
flags.DEFINE_enum('representation', 'extracted', ['extracted', 'pixels', 'pixels_gray', 'simple115v2'],
                  'Definition of the representation used to build the observation. More info in gfootball.env.__init__')
flags.DEFINE_boolean('stacked', True, 'If True, stack 4 observations, otherwise, only the last observation is returned '
                                      'by the environment. Stacking is only possible when representation is one of the '
                                      'following: "pixels", "pixels_gray" or "extracted"')


def main(_):
    def get_run_name() -> str:
        return "EVALUATOR_{algorithm}_scenario-{scenario}_accuracy-{accuracy}".format(
            algorithm=FLAGS.algorithm,
            scenario=FLAGS.scenario_number,
            accuracy=str(FLAGS.accuracy),
        )

    if FLAGS.path is None:
        raise ValueError("path to trained model must be given. Please run script with --help option")
    logger.get_absl_handler().use_absl_log_file(get_run_name(), './')
    from stablebaselines_based.algorithm import load_model
    model = load_model(FLAGS.path, FLAGS.algorithm)
    from stablebaselines_based.evaluator import evaluate_model
    total_reward = evaluate_model(
        model=model,
        scenario_number=FLAGS.scenario_number,
        accuracy=FLAGS.accuracy,
        reward=FLAGS.reward,
        representation=FLAGS.representation,
        stacked=FLAGS.stacked,
        render=FLAGS.render,
        logging=logger.info,
    )
    logger.info(
        "EVALUATOR: ended with {runs} runs and receive in total reward: {reward}. Average reward: {average}".format(
            runs=FLAGS.accuracy,
            reward=total_reward,
            average=total_reward / FLAGS.accuracy,
        ))


if __name__ == '__main__':
    app.run(main)
