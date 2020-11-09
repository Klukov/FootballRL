from stable_baselines import DQN

from rl_project.environment import create_training_env


def learn_dqn(
        env=None,
        policy='CnnPolicy',
        total_number_of_steps=int(1e5),
        learning_rate=5e-4,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        learning_starts=1000,
) -> DQN:
    """
    Parameter's default values are taken from stable_baselines.deepq.dqn.py
    There are some memory problems and you cannot use it for more than 60k time steps with 8GB of Ram
    """
    if env is None:
        env = create_training_env(1)
    dqn = DQN(
        policy=policy,
        env=env,
        gamma=0.99,
        learning_rate=learning_rate,
        buffer_size=50000,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        exploration_initial_eps=exploration_initial_eps,
        train_freq=1,
        batch_size=32,
        double_q=True,
        learning_starts=learning_starts,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        verbose=2,
    )
    return dqn.learn(total_number_of_steps)
