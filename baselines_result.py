from stable_baselines import PPO2
from rl_project.environment import create_demo_env

if __name__ == '__main__':
    model = PPO2.load("trained_models/ppo/ppo_test_model_1M_steps_1-env")
    env = create_demo_env()

    index = 0
    while True:
        index = index + 1
        action, _states = model.predict(env)
        obs, rewards, dones, info = env.step(action)
        if index == 10:
            print("Test: " + info)
            index = 0
