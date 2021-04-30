import os
import numpy as np
import tensorflow as tf
from datetime import  datetime as dt
from envs.qfb_nonlinear_env import QFBNLEnv
from NAF2.naf2 import NAF2

if __name__ == '__main__':
    random_seed = 123
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_name = f"NAF2_QFBNL_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"

    env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
                      calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'))

    env = QFBNLEnv(**env_kwargs)
    eval_env = QFBNLEnv(**env_kwargs)

    training_info = dict(polyak=0.999,
                         batch_size=100,
                         steps_per_batch=10,
                         epochs=1,
                         learning_rate=1e-3,
                         discount=0.9999)
    nafnet_info = dict(hidden_sizes=[100, 100],
                       activation=tf.nn.tanh,
                       kernel_initializer=tf.random_normal_initializer(0, 0.05, seed=random_seed))
    eval_info = dict(eval_env=eval_env,
                     frequency=100,
                     nb_episodes=3,
                     max_ep_steps=50)

    # linearly decaying noise function
    noise_episode_thresh = 40
    noise_fn = lambda act, i: act + np.random.randn(n_act) * max(1 - i / noise_episode_thresh, 0)
    agent = NAF2(env=env,
                 buffer_size=int(5e3),
                 train_every=1,
                 training_info=training_info,
                 eval_info=eval_info,
                 save_frequency=1000,
                 log_frequency=10,
                 directory=os.path.join('models', model_name),
                 tb_log=os.path.join('logs', model_name),
                 # q_smoothing_sigma=0.02,
                 q_smoothing_sigma=0.02,
                 q_smoothing_clip=0.05,
                 nafnet_info=nafnet_info,
                 noise_fn=noise_fn)

    try:
        agent.training(nb_steps=int(5e3 + 1), max_ep_steps=50, warm_up_steps=200, initial_episode_length=5)
    except KeyboardInterrupt:
        print('exiting')