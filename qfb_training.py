import os
import numpy as np

import tensorflow as tf
from stable_baselines import SAC, TD3, PPO2
from NAF2.naf2 import NAF2
from stable_baselines.sac.policies import MlpPolicy as SACPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3Policy
from stable_baselines.common.policies import MlpPolicy as PPOPolicy
from stable_baselines.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines.common.noise import NormalActionNoise, ActionNoise
from datetime import datetime as dt

from qfb_env.qfb_env.qfb_nonlinear_env import QFBNLEnv
# from qfb_env.qfb_nonlinear_env import QFBNLEnv
# from qfb_env.qfb_env import QFBEnv

from replay_buffer.generalizing_replay_wrapper import GeneralizingReplayWrapper

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

class EvaluationAndCheckpointCallback(BaseCallback):
	MAX_EPS = 20
	def __init__(self, env, save_dir, EVAL_FREQ=100, CHKPT_FREQ=1000):
		self.env = env
		self.save_dir = save_dir
		self.model_name = os.path.split(save_dir)[-1]

		self.EVAL_FREQ = EVAL_FREQ
		self.CHKPT_FREQ = CHKPT_FREQ

		self.current_best_model_ep_len = self.env.EPISODE_LENGTH_LIMIT
		self.current_best_model_save_dir = ''

		self.gamma = 0.99
		self.discounts = [np.power(self.gamma, i) for i in range(self.env.EPISODE_LENGTH_LIMIT)]
		super(EvaluationAndCheckpointCallback, self).__init__()

	def quick_save(self, suffix=None):
		if suffix is None:
			save_path = os.path.join(self.save_dir, f'{self.model_name}_{self.num_timesteps}_steps')
		else:
			save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix}')

		self.model.save(save_path)
		if self.verbose > 0:
			print(f'Model saved to: {save_path}')

	def _on_step(self):
		if self.num_timesteps % self.EVAL_FREQ == 0:
			returns = []
			ep_lens = []
			success = []

			### START OF EPISODE LOOP ###
			for ep in range(self.MAX_EPS):
				o = self.env.reset()

				ep_return = 0.0
				step = 0
				d = False
				while not d:
					a = self.model.predict(o, deterministic=True)[0]
					step += 1

					o, r, d, _ = self.env.step(a)

					ep_return += r

				ep_lens.append(step)
				returns.append(ep_return/self.env.REWARD_SCALE)
				if step < self.env.max_steps:
					success.append(1.0)
				else:
					success.append(0.0)
			### END OF EPISODE LOOP ###

			returns = np.mean(returns)
			ep_lens = np.mean(ep_lens)
			success = np.mean(success) * 100

			### SAVE SUCCESSFUL AGENTS ###
			if success > 0:
				self.quick_save()
				if self.verbose > 1:
					print("Saving model checkpoint to {}".format(path))

			for tag, val in zip(('episode_return', 'episode_length', 'success'),
								(returns, ep_lens, success)):
				summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
				self.locals['writer'].add_summary(summary, self.num_timesteps)

		if self.num_timesteps % self.CHKPT_FREQ == 0:
			self.quick_save()

		return True

class DecayingNormalActionNoise(ActionNoise):
	def __init__(self, n_act, eps_thresh):
		self.eps_thresh = eps_thresh
		self.cur_ep = 0
		# self.n_act = n_act

	def reset(self) -> None:
		self.cur_ep += 1

	def __call__(self):
		# return np.random.randn(self.n_act) * max(1 - self.cur_ep / self.eps_thresh, 0)
		return np.random.randn() * max(1 - self.cur_ep / self.eps_thresh, 0)

	def __repr__(self):
		return f'{type(self).__name__}__eps-thresh={self.eps_thresh}'

class DecayingLearningRate():
	def __init__(self, init_lr, final_lr, frac_decay):
		self.init_lr = init_lr
		self.final_lr = final_lr
		self.frac_decay = frac_decay

	def __call__(self, frac):
		return self.init_lr - (self.init_lr - self.final_lr) * min(((1-frac)/self.frac_decay), 1)

	def __repr__(self):
		return f'{type(self).__name__}__init-lr={self.init_lr}__final-lr={self.final_lr}__frac-decay={self.frac_decay}'

def train_random_seed():
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	for random_seed in (123, 234, 345, 456, 567):
	# for random_seed in (234, 345, 456, 567):
		model_name = f"SAC_QFBNL_{dt.strftime(dt.now(), '%m%d%y_%H%M%S')}"
		print(model_name)
		save_dir = os.path.join('models', model_name)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		np.random.seed(random_seed)

		env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
						  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
						  perturb_state=False,
						  noise_std=0.0)

		env = QFBNLEnv(**env_kwargs)
		eval_env = QFBNLEnv(**env_kwargs)

		nb_steps = int(3e3)

		# action_noise = DecayingNormalActionNoise(n_act=env.act_dimension, eps_thresh=5000)
		#
		#
		# lr_decay_steps = int(6e4)
		# lr_decay_frac = lr_decay_steps / nb_steps
		# learning_rate = DecayingLearningRate(5e-5, 5e-6, lr_decay_frac)

		##### SAC start #####
		sac_params = dict(
			gamma = 0.99,
			learning_rate = 1e-3,
			buffer_size = int(5e5),#nb_steps,
			learning_starts = 1000,
			train_freq = 1,
			batch_size = 256,#64,
			tau = 0.005,
			ent_coef = 'auto',
			target_update_interval = 1,
			gradient_steps = 3,
			target_entropy = 'auto',
			action_noise = None,
			random_exploration = 0.0,
			policy_kwargs={'layers': [50, 50]}
		)
		replay_wrapper = None#GeneralizingReplayWrapper
		model = SAC(SACPolicy, env, **sac_params, verbose=1, tensorboard_log='logs',
					_init_setup_model=True, full_tensorboard_log=False,
					seed=random_seed, n_cpu_tf_sess=6)
		##### SAC finish #####

		##### PPO2 start #####
		# ppo_params = dict(
		# 	gamma = 0.99,
		# 	n_steps = 128,
		# 	ent_coef = 0.01,
		# 	learning_rate = 2.5e-4,
		# 	vf_coef = 0.5,
		# 	max_grad_norm = 0.5,
		# 	lam = 0.95,
		# 	nminibatches = 4,
		# 	noptepochs = 4,
		# 	cliprange = 0.2,
		# 	cliprange_vf = None,
		# 	policy_kwargs = {'net_arch': [100, 100]}
		# )
		# model = PPO2(PPOPolicy, env, **ppo_params, verbose=1,
		# 			 tensorboard_log='logs', _init_setup_model=True,
		# 			 full_tensorboard_log=False, seed=random_seed, n_cpu_tf_sess=None)
		##### PPO2 finish #####

		##### TD3 start #####
		# td3_params = dict(
		# 	gamma=0.99,
		# 	# learning_rate=3e-4, # default
		# 	learning_rate=learning_rate,
		# 	# buffer_size=50000, # default
		# 	buffer_size=10000,
		# 	learning_starts=100,
		# 	train_freq=100,
		# 	gradient_steps=100,
		# 	# batch_size=64, # default
		# 	batch_size=128, # default
		# 	# tau=0.005, # default
		# 	tau=0.05, # default
		# 	# policy_delay=2, # default
		# 	policy_delay=5,
		# 	action_noise=action_noise,
		# 	target_policy_noise=0.2, # default
		# 	target_noise_clip=0.5,   # default
		# 	# target_policy_noise=0.02,
		# 	# target_noise_clip=0.05,
		# 	random_exploration=0.0
		# )
		# td3_policy_kwargs = {'layers': [50, 50]}
		# replay_wrapper = None
		# model = TD3(TD3Policy, env, **td3_params, verbose=1, tensorboard_log='logs',
		# 			_init_setup_model=True, policy_kwargs=td3_policy_kwargs, full_tensorboard_log=False,
		# 			seed=random_seed, n_cpu_tf_sess=None)

		##### TD3 finish #####

		# callback_chkpt = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix=model_name)
		eval_callback = EvaluationAndCheckpointCallback(eval_env, save_dir=save_dir,
														EVAL_FREQ=100, CHKPT_FREQ=1000)

		# callbacks = [callback_chkpt, eval_callback]
		callbacks = eval_callback

		# Save parameters to save_logs.txt
		with open(os.path.join('models', 'save_logs.txt'), 'a') as f:
			f.write(f'\n-> {save_dir}\n')

			if isinstance(model, SAC):
				params = sac_params
			elif isinstance(model, PPO2):
				params = ppo_params
			elif isinstance(model, TD3):
				params = td3_params
				params['action_noise'] = action_noise
			params['random_seed'] = random_seed

			for param_name, param_val in params.items():
				f.write(f' `-> {param_name} = {param_val}\n')

		# Start learning
		if isinstance(model, SAC):
			model.learn(total_timesteps=nb_steps, log_interval=10, replay_wrapper=replay_wrapper,
						tb_log_name=model_name, callback=callbacks)
		elif isinstance(model, PPO2):
			model.learn(total_timesteps=nb_steps, log_interval=10, reset_num_timesteps=True,
						tb_log_name=model_name, callback=callbacks)
		elif isinstance(model, TD3):
			model.learn(total_timesteps=nb_steps, log_interval=10,  reset_num_timesteps=True,
						tb_log_name=model_name, callback=callbacks,
						replay_wrapper=replay_wrapper)

		# Save latest model
		save_path = os.path.join(save_dir, f'{model_name}_final')
		model.save(save_path)
		print(f'Model saved to: {save_path}')
		# break

def train_env_reward_scale():
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	random_seed = 123
	for reward_scale in (0.01, 0.1, 1, 10, 100):
		model_name = f"SAC_QFBNL_{dt.strftime(dt.now(), '%m%d%y_%H%M%S')}"
		print(model_name)
		save_dir = os.path.join('models', model_name)

		np.random.seed(random_seed)

		env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
						  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
						  perturb_state=False,
						  noise_std=0.0)

		env = QFBNLEnv(**env_kwargs)
		eval_env = QFBNLEnv(**env_kwargs)

		for e in (env, eval_env):
			e.reward_scale = reward_scale

		nb_steps = int(2e5)

		# action_noise = DecayingNormalActionNoise(n_act=env.act_dimension, eps_thresh=5000)
		#
		#
		# lr_decay_steps = int(6e4)
		# lr_decay_frac = lr_decay_steps / nb_steps
		# learning_rate = DecayingLearningRate(5e-5, 5e-6, lr_decay_frac)

		##### SAC start #####
		sac_params = dict(
			gamma = 0.99,
			learning_rate = 3e-4,
			buffer_size = nb_steps,
			learning_starts = 100,
			train_freq = 1,
			batch_size = 256,#64,
			tau = 0.005,
			ent_coef = 'auto',
			target_update_interval = 1,
			gradient_steps = 1,
			target_entropy = 'auto',
			action_noise = None,
			random_exploration = 0.1,
			policy_kwargs={'layers': [50, 50]}
		)
		replay_wrapper = None#GeneralizingReplayWrapper
		model = SAC(SACPolicy, env, **sac_params, verbose=1, tensorboard_log='logs',
					_init_setup_model=True, full_tensorboard_log=False,
					seed=random_seed, n_cpu_tf_sess=6)
		##### SAC finish #####

		##### PPO2 start #####
		# ppo_params = dict(
		# 	gamma = 0.99,
		# 	n_steps = 128,
		# 	ent_coef = 0.01,
		# 	learning_rate = 2.5e-4,
		# 	vf_coef = 0.5,
		# 	max_grad_norm = 0.5,
		# 	lam = 0.95,
		# 	nminibatches = 4,
		# 	noptepochs = 4,
		# 	cliprange = 0.2,
		# 	cliprange_vf = None,
		# 	policy_kwargs = {'net_arch': [100, 100]}
		# )
		# model = PPO2(PPOPolicy, env, **ppo_params, verbose=1,
		# 			 tensorboard_log='logs', _init_setup_model=True,
		# 			 full_tensorboard_log=False, seed=random_seed, n_cpu_tf_sess=None)
		##### PPO2 finish #####

		##### TD3 start #####
		# td3_params = dict(
		# 	gamma=0.99,
		# 	# learning_rate=3e-4, # default
		# 	learning_rate=learning_rate,
		# 	# buffer_size=50000, # default
		# 	buffer_size=10000,
		# 	learning_starts=100,
		# 	train_freq=100,
		# 	gradient_steps=100,
		# 	# batch_size=64, # default
		# 	batch_size=128, # default
		# 	# tau=0.005, # default
		# 	tau=0.05, # default
		# 	# policy_delay=2, # default
		# 	policy_delay=5,
		# 	action_noise=action_noise,
		# 	target_policy_noise=0.2, # default
		# 	target_noise_clip=0.5,   # default
		# 	# target_policy_noise=0.02,
		# 	# target_noise_clip=0.05,
		# 	random_exploration=0.0
		# )
		# td3_policy_kwargs = {'layers': [50, 50]}
		# replay_wrapper = None
		# model = TD3(TD3Policy, env, **td3_params, verbose=1, tensorboard_log='logs',
		# 			_init_setup_model=True, policy_kwargs=td3_policy_kwargs, full_tensorboard_log=False,
		# 			seed=random_seed, n_cpu_tf_sess=None)

		##### TD3 finish #####

		callback_chkpt = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix=model_name)
		eval_callback = EvaluationAndCheckpointCallback(eval_env)

		# Save parameters to save_logs.txt
		with open(os.path.join('models', 'save_logs.txt'), 'a') as f:
			f.write(f'\n-> {save_dir}\n')

			if isinstance(model, SAC):
				params = sac_params
			elif isinstance(model, PPO2):
				params = ppo_params
			elif isinstance(model, TD3):
				params = td3_params
				params['action_noise'] = action_noise
			params['random_seed'] = random_seed
			params['reward_scale'] = reward_scale

			for param_name, param_val in params.items():
				f.write(f' `-> {param_name} = {param_val}\n')

		# Start learning
		if isinstance(model, SAC):
			model.learn(total_timesteps=nb_steps, log_interval=10, replay_wrapper=replay_wrapper,
						tb_log_name=model_name, callback=[callback_chkpt, eval_callback])
		elif isinstance(model, PPO2):
			model.learn(total_timesteps=nb_steps, log_interval=10, reset_num_timesteps=True,
						tb_log_name=model_name, callback=[callback_chkpt, eval_callback])
		elif isinstance(model, TD3):
			model.learn(total_timesteps=nb_steps, log_interval=10,  reset_num_timesteps=True,
						tb_log_name=model_name, callback=[callback_chkpt, eval_callback],
						replay_wrapper=replay_wrapper)

		# Save latest model
		save_path = os.path.join('models', model_name, f'{model_name}_final')
		model.save(save_path)
		print(f'Model saved to: {save_dir}')
		# break

def train_hparams():
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	random_seed = 123

	par_dir = 'models_td3_hparams'

	# sac_params_default = dict(
	# 	gamma=0.99,
	# 	learning_rate=3e-4,
	# 	buffer_size=50000,
	# 	learning_starts=100,
	# 	train_freq=1,
	# 	batch_size=64,
	# 	tau=0.005,
	# 	ent_coef='auto',
	# 	target_update_interval=1,
	# 	gradient_steps=1,
	# 	target_entropy='auto',
	# 	action_noise=None,
	# 	random_exploration=0.0
	# )
	td3_params_default = dict(
		gamma=0.99,
		learning_rate=3e-4,
		buffer_size=50000,
		learning_starts=100,
		train_freq=100,
		gradient_steps=100,
		batch_size=128,
		tau=0.005,
		policy_delay=2,
		action_noise=None,
		target_policy_noise=0.2,
		target_noise_clip=0.5,
		random_exploration=0.0
	)

	td3_params_list = []

	for lr in [3e-4, 3e-5]:
		for buff in [1000, 10000, 50000]:
			for batch in [32, 64, 128, 512]:
				for tau in [0.05, 0.005]:
					td3_params_list.append(td3_params_default.copy())
					td3_params_list[-1]['learning_rate'] = lr
					td3_params_list[-1]['buffer_size'] = buff
					td3_params_list[-1]['batch_size'] = batch
					td3_params_list[-1]['tau'] = tau

	for td3_params in td3_params_list:
		model_name = f"TD3_QFBNL_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"
		save_dir = os.path.join(par_dir, model_name)
		# if not os.path.exists(save_dir):
		# 	os.makedirs(save_dir)

		# random_seed = 123
		np.random.seed(random_seed)

		env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
						  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
						  perturb_state=False,
						  noise_std=0.0)

		env = QFBNLEnv(**env_kwargs)
		eval_env = QFBNLEnv(**env_kwargs)

		# action_noise = NormalActionNoise(mean=0.0, sigma=0.1)

		# lr_initial = 1e-4
		# lr_final = 1e-5
		# lr_linear_steps = int(8e4)
		# lr_fn = lambda f: lr_initial - min((f)*nb_steps/lr_linear_steps, 1)*(lr_initial - lr_final)\\\\\\\\\\\\\\\\\\\\

		nb_steps = int(8e4)

		##### SAC start #####
		# sac_policy_kwargs = {'layers': [50, 50]}
		# replay_wrapper = None  # GeneralizingReplayWrapper
		# model = SAC(SACPolicy, env, **sac_params, verbose=1, tensorboard_log=os.path.join(par_dir, 'logs'),
		# 			_init_setup_model=True, policy_kwargs=sac_policy_kwargs, full_tensorboard_log=False,
		# 			seed=random_seed, n_cpu_tf_sess=7)
		##### SAC finish #####

		##### PPO2 start #####
		# ppo_params = dict(
		# 	gamma = 0.99,
		# 	n_steps = 128,
		# 	ent_coef = 0.01,
		# 	learning_rate = 2.5e-4,
		# 	vf_coef = 0.5,
		# 	max_grad_norm = 0.5,
		# 	lam = 0.95,
		# 	nminibatches = 4,
		# 	noptepochs = 4,
		# 	cliprange = 0.2,
		# 	cliprange_vf = None
		# )
		# ppo_policy_kwargs = {'layers': [50, 50]}
		# model = PPO2(PPOPolicy, env, **ppo_params, verbose=1,
		# 			 tensorboard_log='logs', _init_setup_model=True, policy_kwargs=ppo_policy_kwargs,
		# 			 full_tensorboard_log=False, seed=random_seed, n_cpu_tf_sess=None)
		##### PPO2 finish #####

		##### TD3 start #####
		td3_policy_kwargs = {'layers': [50, 50]}
		replay_wrapper = None
		model = TD3(TD3Policy, env, **td3_params, verbose=1, tensorboard_log=os.path.join(par_dir, 'logs'),
					_init_setup_model=True, policy_kwargs=td3_policy_kwargs, full_tensorboard_log=False,
					seed=random_seed, n_cpu_tf_sess=None)

		##### TD3 finish #####

		callback_chkpt = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix=model_name)
		eval_callback = EvaluationAndCheckpointCallback(eval_env)

		# Save parameters to save_logs.txt
		with open(os.path.join(par_dir, 'save_logs.txt'), 'a') as f:
			f.write(f'\n-> {save_dir}\n')

			if isinstance(model, SAC):
				params = sac_params
				params['policy_kwargs'] = sac_policy_kwargs
			elif isinstance(model, PPO2):
				params = ppo_params
				params['policy_kwargs'] = ppo_policy_kwargs
			elif isinstance(model, TD3):
				params = td3_params
				params['policy_kwargs'] = td3_policy_kwargs
			params['random_seed'] = random_seed

			for param_name, param_val in params.items():
				f.write(f' `-> {param_name} = {param_val}\n')

		# Start learning
		if isinstance(model, SAC):
			model.learn(total_timesteps=nb_steps, log_interval=10, replay_wrapper=replay_wrapper,
						tb_log_name=model_name, callback=[callback_chkpt, eval_callback])
		elif isinstance(model, PPO2):
			model.learn(total_timesteps=nb_steps, log_interval=10, reset_num_timesteps=True,
						tb_log_name=model_name, callback=[callback_chkpt, eval_callback])
		elif isinstance(model, TD3):
			model.learn(total_timesteps=nb_steps, log_interval=10, reset_num_timesteps=True,
						tb_log_name=model_name, callback=[callback_chkpt, eval_callback],
						replay_wrapper=replay_wrapper)

		# Save latest model
		save_path = os.path.join(par_dir, model_name, f'{model_name}_final')
		model.save(save_path)
		print(f'Model saved to: {save_dir}')
answer=-1/12
if __name__ == '__main__':
    # train_hparams()
	train_random_seed()
	# train_env_reward_scale()