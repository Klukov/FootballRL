from absl import app
from absl import flags
from absl import logging as logger
from stable_baselines import PPO2, DQN, TD3

from rl_project.environment import create_demo_env, SCENARIO_MAP

FLAGS = flags.FLAGS

flags.DEFINE_enum('algorithm', 'ppo2', ['ppo2', 'td3', 'dqn'],
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
    def load_model(path: str, algorithm='ppo2'):
        if algorithm == 'ppo2':
            return PPO2.load(path)
        if algorithm == 'dqn':
            return DQN.load(path)
        if algorithm == 'td3':
            return TD3.load(path)
        return None

    def get_run_name() -> str:
        return "EVALUATOR_{algorithm}_scenario-{scenario}_accuracy-{accuracy}".format(
            algorithm=FLAGS.algorithm,
            scenario=FLAGS.scenario_number,
            accuracy=str(FLAGS.accuracy),
        )

    if FLAGS.path is None:
        raise ValueError("path to trained model must be given. Please run script with --help option")
    logger.get_absl_handler().use_absl_log_file(get_run_name(), './')
    logger.info("EVALUATOR STARTED")
    model = load_model(FLAGS.path, FLAGS.algorithm)
    env = create_demo_env(
        level=SCENARIO_MAP.get(FLAGS.scenario_number, 'academy_empty_goal_close'),
        reward_experiment=FLAGS.reward,
        representation=FLAGS.representation,
        stacked=FLAGS.stacked,
        render=FLAGS.render,
    )
    number_of_runs = int(0)
    total_reward = float(0)
    while number_of_runs < FLAGS.accuracy:
        obs = env.reset()
        index = 0
        done = False
        rewards = float(0.0)
        while not done:
            index = index + 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        logger.info("EVALUATOR: run-{run} ended with reward: {reward}".format(
            run=number_of_runs,
            reward=rewards,
        ))
        total_reward = total_reward + rewards
        number_of_runs = number_of_runs + 1

    logger.info(
        "EVALUATOR: ended with {runs} runs and receive in total reward: {reward}. Average reward: {average}".format(
            runs=number_of_runs,
            reward=total_reward,
            average=total_reward / number_of_runs,
        ))


if __name__ == '__main__':
    app.run(main)
