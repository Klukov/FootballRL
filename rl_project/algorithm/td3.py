from stable_baselines import TD3

from rl_project.environment import create_training_env


def learn_td3(
        env=None,
        policy='CnnPolicy',
        total_number_of_steps=int(1e5),
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        train_freq=100,
        gradient_steps=100,
        batch_size=128,
        tau=0.005,
) -> TD3:
    """
    Parameter's default values are taken from stable_baselines.td3.td3.py
    The exception is learning_starts which was set from 100 to 1000
    """
    if env is None:
        env = create_training_env(1)
    td3 = TD3(
        policy=policy,
        env=env,
        gamma=0.99,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        batch_size=batch_size,
        tau=tau,
        policy_delay=2,
        action_noise=None,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        random_exploration=0.0,
    )
    return td3.learn(total_number_of_steps)
