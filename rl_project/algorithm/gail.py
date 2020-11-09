from stable_baselines import GAIL

from rl_project.environment import create_training_env


def learn_gail(
        vec_env=None,
        policy='CnnPolicy',
        total_number_of_steps=int(1e5),
) -> GAIL:
    """
    Parameter's default values are taken from stable_baselines.gail.model.py
    and because GAIL is extension of TRPO then rest of parameters are taken
    from stable_baselines.trpo_mpi.trpo_mpi.py
    """
    if vec_env is None:
        vec_env = create_training_env(1)
    gail = GAIL(
        policy=policy,
        env=vec_env,
        hidden_size_adversary=100,
        adversary_entcoeff=1e-3,
        g_step=3,
        d_step=1,
        d_stepsize=3e-4,
        verbose=2,
        gamma=0.99,
        timesteps_per_batch=1024,
        max_kl=0.01,
        cg_iters=10,
        lam=0.98,
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters=3
    )
    return gail.learn(total_number_of_steps)
