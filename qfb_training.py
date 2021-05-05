import os
import numpy as np

import tensorflow as tf
from stable_baselines import SAC, TD3
from stable_baselines.sac.policies import MlpPolicy as SACPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3Policy
from stable_baselines.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines.common.noise import NormalActionNoise
from datetime import datetime as dt

from envs.qfb_nonlinear_env import QFBNLEnv
from envs.qfb_env import QFBEnv

from replay_buffer.generalizing_replay_wrapper import GeneralizingReplayWrapper

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

class EvaluationCallback(BaseCallback):
	MAX_EPS = 200
	def __init__(self, env):
		self.env = env
		self.current_best_model_ep_len = self.env.EPISODE_LENGTH_LIMIT
		self.current_best_model_save_dir = ''
		self.commited_returns = np.array([])
		self.uncommitted_returns = np.array([])
		super(EvaluationCallback, self).__init__()

	def _on_step(self):
		if self.num_timesteps % 1000 == 0:
			returns = []
			ep_lens = []
			success = []
			for ep in range(self.MAX_EPS):
				o = self.env.reset()
				ret = 0
				step = -1
				while True:
					step += 1
					a = self.model.predict(o)[0]
					o, r, d, _ = self.env.step(a)
					ret += r
					if d:
						break
				ep_lens.append(step + 1)
				returns.append(ret)
				if ep_lens[-1] == self.env.EPISODE_LENGTH_LIMIT:
					success.append(0.0)
				else:
					success.append(1.0)
			returns = np.mean(returns)
			ep_lens = np.mean(ep_lens)
			success = np.mean(success) * 100
			self.uncommitted_returns = np.append(self.uncommitted_returns, returns)

			summary1 = tf.Summary(value=[tf.Summary.Value(tag='episode_return', simple_value=returns)])
			summary2 = tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=ep_lens)])
			summary3 = tf.Summary(value=[tf.Summary.Value(tag='success', simple_value=success)])
			self.locals['writer'].add_summary(summary1, self.num_timesteps)
			self.locals['writer'].add_summary(summary2, self.num_timesteps)
			self.locals['writer'].add_summary(summary3, self.num_timesteps)

			if ep_lens < self.current_best_model_ep_len:
				self.current_best_model_save_dir = save_path = os.path.join('models', model_name, f'{model_name}_step{self.num_timesteps}')
				self.model.save(save_path)
				print(f'-> Current best model has average episode length: {ep_lens}\n' \
					  f' `-> {((self.current_best_model_ep_len - ep_lens) / self.current_best_model_ep_len) * 100.0:.2f}% improvement\n' \
					  f' `-> {success}% successful'\
					  f' `-> Model saved to: {save_path}')

				self.current_best_model_ep_len = ep_lens
				self.commited_returns = np.concatenate([self.commited_returns, self.uncommitted_returns])
				self.uncommitted_returns = np.array([])

				self.training_env.noise_std *= 0.9
				self.env.noise_std *= 0.9

		return True

# class SACHyperparameters(SAC):
# 	def __init__(self, policy, env, learning_rate, **kwargs):
# 		super(SACHyperparameters, self).__init__(policy, env, learning_rate, **kwargs)
# 		self.learning_rate
# 		self.hparams = kwargs
# 		self.eval_callback = EvaluationCallback(eval_env)
# 		self.relax_coeff = 0.9
# 		self.stress_coeff = 1.1
#
# 	def relax(self):
# 		self.scale(self.relax_coeff)
#
# 	def stress(self):
# 		self.scale(self.stress_coeff)
#
# 	def scale(self, s):
# 		self.training_env.noise_std *= s
# 		self.hparams['learning_rate'] *= s
#
# 	def learn(self, total_timesteps, **kwargs):
# 		super(SACHyperparameters, self).learn(total_timesteps, **kwargs)




if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	model_name = f"SAC_QFBNL_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"

	env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
					  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
					  perturb_state=False,
					  noise_std=0.0)

	env = QFBNLEnv(**env_kwargs)
	eval_env = QFBNLEnv(**env_kwargs)

	# action_noise = NormalActionNoise(mean=0.0, sigma=0.1)
	# model = TD3(TD3Policy, env, gamma=0.99, learning_rate=1e-4, buffer_size=100000, learning_starts=100,
	# 			train_freq=100, gradient_steps=100, batch_size=128, tau=0.005, policy_delay=2,
	# 			action_noise=action_noise, target_policy_noise=0.2, target_noise_clip=0.5, random_exploration=0.0,
	# 			verbose=1, tensorboard_log='logs',
	# 			_init_setup_model=True, policy_kwargs={'layers': [100, 100]},
	# 			full_tensorboard_log=False, seed=0, n_cpu_tf_sess=None)

	# model = SAC(SACPolicy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
	#              learning_starts=100, train_freq=1, batch_size=64,
	#              tau=0.005, ent_coef='auto', target_update_interval=1,
	#              gradient_steps=1, target_entropy='auto', action_noise=None,
	#              random_exploration=0.0, verbose=1, tensorboard_log='logs',
	#              _init_setup_model=True, policy_kwargs={'layers':[400, 200]}, full_tensorboard_log=True,
	#              seed=None, n_cpu_tf_sess=None)
	# lr_initial = 1e-4
	# lr_final = 1e-5
	nb_steps = int(5e6)
	# lr_linear_steps = int(8e4)
	# lr_fn = lambda f: lr_initial - min((f)*nb_steps/lr_linear_steps, 1)*(lr_initial - lr_final)
	learning_rate = 3e-4
	buffer_size = nb_steps
	learning_starts = 100
	batch_size = 256
	policy_kwargs = {'layers': [256, 256]}
	random_seed = 123
	replay_wrapper = GeneralizingReplayWrapper

	np.random.seed(random_seed)
	model = SAC(SACPolicy, env, gamma=0.99, learning_rate=learning_rate, buffer_size=buffer_size,
				learning_starts=learning_starts, train_freq=1, batch_size=batch_size,
				tau=0.005, ent_coef='auto', target_update_interval=1,
				gradient_steps=1, target_entropy='auto', action_noise=None,
				random_exploration=0.0, verbose=1, tensorboard_log='logs',
				_init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=False,
				seed=random_seed, n_cpu_tf_sess=None)

	save_dir = os.path.join('models', model_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# callback_chkpt = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix=model_name)
	eval_callback = EvaluationCallback(eval_env)


	with open(os.path.join('models', 'save_logs.txt'), 'a') as f:
		f.write(f'-> {save_dir}\n')
		for param_name in ('nb_steps', 'learning_rate', 'buffer_size', 'learning_starts', 'batch_size',
						   'policy_kwargs', 'random_seed', 'replay_wrapper'):
			f.write(f' `-> {param_name} = {locals()[param_name]}\n')

	model.learn(total_timesteps=nb_steps, log_interval=1000, replay_wrapper=replay_wrapper,
				tb_log_name=model_name, callback=eval_callback)

	save_path = os.path.join('models', model_name, f'{model_name}_final')
	model.save(save_path)
	print(f'Model saved to: {save_dir}')