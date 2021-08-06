import os

import tensorflow as tf
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as SACPolicy
from stable_baselines.common.callbacks import CheckpointCallback
from datetime import datetime as dt

# from qfb_env.qfb_env import QFBEnv
from qfb_env.qfb_env_carnival import QFBEnvCarnival

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	model_name = f"SAC_QFB_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"
	env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
					 calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'))
	env = QFBEnvCarnival(noise_std=0.0, **env_kwargs)
	eval_env = QFBEnvCarnival(noise_std=0.0, **env_kwargs)

	model = SAC(SACPolicy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=1, tensorboard_log='logs',
                 _init_setup_model=True, policy_kwargs={'layers':[400, 200]}, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None)

	callback_chkpt = CheckpointCallback(save_freq=2000, save_path='models', name_prefix=model_name)

	nb_steps = int(1e6)
	# MyReplayBuffer.setup_storage(size=nb_steps, n_obs=env.obs_dimension, n_act=env.act_dimension)

	model.learn(total_timesteps=nb_steps, log_interval=200, #replay_wrapper=MyReplayBuffer,
				tb_log_name=model_name, callback=callback_chkpt)

	# MyReplayBuffer.save_storage('buffer_storage_tmp.pkl')

	save_path = os.path.join('models', model_name)
	model.save(save_path)
	print(f'Model saved to: {save_path}')