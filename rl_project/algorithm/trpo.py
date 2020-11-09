from stable_baselines import TRPO

from rl_project.environment import create_training_env


def learn_trpo(
        vec_env=None,
        policy='CnnPolicy',
        total_number_of_steps=int(1e5),
) -> TRPO:
    """
    Parameter's default values are taken from stable_baselines.trpo_mpi.trpo_mpi.py
    """
    if vec_env is None:
        vec_env = create_training_env(1)
    trpo = TRPO(
        policy=policy,
        env=vec_env,
        gamma=0.99,
        timesteps_per_batch=1024,
        max_kl=0.01,
        cg_iters=10,
        lam=0.98,
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters=3,
        verbose=2,
    )
    return trpo.learn(total_number_of_steps)
