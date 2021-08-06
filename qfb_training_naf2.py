import os
import numpy as np
import tensorflow as tf
from datetime import  datetime as dt
from qfb_env.qfb_nonlinear_env import QFBNLEnv
from NAF2.naf2 import NAF2

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_name = f"NAF2_QFBNL_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"
    save_dir = os.path.join('models', model_name)
    log_dir = os.path.join('logs', model_name)

    env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
                      calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
                      perturb_state=False,
                      noise_std=0.0)

    env = QFBNLEnv(**env_kwargs)
    eval_env = QFBNLEnv(**env_kwargs)

    random_seed = 567
    nb_steps = int(1e5)

    training_info = dict(polyak=0.999,
                         batch_size=100,
                         steps_per_batch=10,
                         epochs=1,
                         learning_rate=1e-3,
                         discount=0.9999)
    nafnet_info = dict(hidden_sizes=[50, 50],
                       activation=tf.nn.relu,
                       kernel_initializer=tf.random_normal_initializer(0, 0.05, seed=random_seed))
    eval_info = dict(eval_env=eval_env,
                     frequency=100,
                     nb_episodes=20)
    params = dict(buffer_size=int(5e3),
                  q_smoothing_sigma=0.02,
                  q_smoothing_clip=0.05)

    # linearly decaying noise function
    noise_episode_thresh = 40
    n_act = env.act_dimension
    noise_fn = lambda act, i: act + np.random.randn(n_act) * max(1 - i / noise_episode_thresh, 0)
    agent = NAF2(env=env,
                 buffer_size=params['buffer_size'],
                 train_every=1,
                 training_info=training_info,
                 eval_info=eval_info,
                 save_frequency=200,
                 log_frequency=10,
                 directory=save_dir,
                 tb_log=log_dir,
                 q_smoothing_sigma=params['q_smoothing_sigma'],
                 q_smoothing_clip=params['q_smoothing_clip'],
                 nafnet_info=nafnet_info,
                 noise_fn=noise_fn)

    with open(os.path.join('models', 'save_logs.txt'), 'a') as f:
        f.write(f'\n-> {save_dir}\n')

        all_params = {**training_info, **nafnet_info, **params}
        all_params['noise_fn'] = f'lambda act, i: act + np.random.randn({n_act}) * max(1 - i / {noise_episode_thresh}, 0)'
        all_params['random_seed'] = random_seed

        for param_name, param_val in all_params.items():
            f.write(f' `-> {param_name} = {param_val}\n')

    try:
        agent.training(nb_steps=nb_steps, max_ep_steps=70, warm_up_steps=200, initial_episode_length=5)
    except KeyboardInterrupt:
        print('exiting')