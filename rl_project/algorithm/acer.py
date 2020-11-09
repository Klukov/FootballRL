from stable_baselines import ACER

from rl_project.environment import create_training_env


def learn_acer(
        vec_env=None,
        policy='CnnPolicy',
        total_number_of_steps=int(1e5),
        learning_rate=7e-4,
        n_steps=20,
        max_grad_norm=10,
        lr_schedule='linear',
        buffer_size=5000,
        replay_start=1000
) -> ACER:
    """
    Parameter's default values are taken from stable_baselines.acer.acer_simple.py
    """
    if vec_env is None:
        vec_env = create_training_env(1)
    acer = ACER(
        policy=policy,
        env=vec_env,
        gamma=0.99,
        n_steps=n_steps,
        num_procs=None,
        q_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        rprop_alpha=0.99,
        rprop_epsilon=1e-5,
        buffer_size=buffer_size,
        replay_ratio=4,
        replay_start=replay_start,
        correction_term=10.0,
        trust_region=True,
        alpha=0.99,
        delta=1,
        verbose=2
    )
    return acer.learn(total_number_of_steps)
