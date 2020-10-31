import multiprocessing

from stable_baselines.ppo2 import PPO2

from rl_project.environment import create_training_env


def learn_ppo2(
        vec_env=None,
        policy='CnnPolicy',
        total_number_of_steps=int(1e5),  # num_timesteps
        seed=0,
        number_of_steps_per_epoch=128,  # nsteps
        number_of_mini_batches_in_epoch=8,  # nminibatches
        number_of_updates_per_epoch=4,  # noptepochs
        max_grad_norm=0.5,
        gamma=0.993,  # discount factor
        entropy_coefficient=0.01,  # ent_coef
        learning_rate=0.00008,  # lr
        clip_range=0.27,  # cliprange
) -> PPO2:
    """
    Parameter's default values are taken from football.gfootball.examples.run_ppo2.py
    """
    if vec_env is None:
        vec_env = create_training_env(1)

    import tensorflow.compat.v1 as tf
    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    ppo2 = PPO2(
        policy=policy,
        env=vec_env,
        gamma=gamma,
        n_steps=number_of_steps_per_epoch,
        ent_coef=entropy_coefficient,
        learning_rate=learning_rate,
        vf_coef=1,
        max_grad_norm=max_grad_norm,
        nminibatches=number_of_mini_batches_in_epoch,
        noptepochs=number_of_updates_per_epoch,
        cliprange=clip_range,
        seed=seed,
    )
    return ppo2.learn(total_timesteps=total_number_of_steps)
