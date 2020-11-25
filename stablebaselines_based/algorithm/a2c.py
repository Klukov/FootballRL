from stable_baselines import A2C

from stablebaselines_based.environment import create_training_env


def get_a2c(
        vec_env=None,
        policy='CnnPolicy',
        learning_rate=7e-4,
        momentum=0.0,
        alpha=0.99,
        epsilon=1e-5,
        max_grad_norm=0.5,
        lr_schedule='constant'
) -> A2C:
    """
    Parameter's default values are taken from stable_baselines.a2c.a2c.py
    """
    if vec_env is None:
        vec_env = create_training_env(1)
    return A2C(
        policy=policy,
        env=vec_env,
        gamma=0.99,
        n_steps=5,
        vf_coef=0.25,
        ent_coef=0.01,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        alpha=alpha,
        momentum=momentum,
        epsilon=epsilon,
        lr_schedule=lr_schedule,
        verbose=2
    )
