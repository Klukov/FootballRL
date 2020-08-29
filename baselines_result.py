from stable_baselines import PPO2
from rl_project.environment import create_demo_env

if __name__ == '__main__':
    model = PPO2.load("trained_models/academy_empty_goal_close/ppo/ppo_test_model_1M_steps_8-env")
    env = create_demo_env()
    while True:
        obs = env.reset()
        index = 0
        done = False
        while not done:
            index = index + 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
