import os

from stable_baselines import TD3, DDPG
from stable_baselines.td3.policies import MlpPolicy as TD3Policy
from datetime import datetime as dt

from qfb_env import QFBEnv
from ReplayBuffer.my_buffer import MyReplayBuffer

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':
	model_name = f"TD3_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"
	env = QFBEnv()

	'''
	Env reward_accumulated_limit=-10
	models/TD_121120_2222:
		model = TD3(TD3Policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000, learning_starts=100,
						train_freq=100, gradient_steps=100, batch_size=128, tau=0.005, policy_delay=2,
						action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5, random_exploration=0.0,
						verbose=1, tensorboard_log='logs', _init_setup_model=True, policy_kwargs={'layers':[100, 100]},
						full_tensorboard_log=False, seed=0, n_cpu_tf_sess=None)
		model.learn(total_timesteps=100000, log_interval=100, tb_log_name=model_name)

	models/TD_121120_2250:
		Added Gaussian noise std=0.05 to each action in QFBEnv.step()
	
	models/TD_121120_2310
		Changed QFBEnv.objective:
		def objective(self):
			state_reward = -np.sum(np.abs(self._current_state))
			action_reward = -np.sum(np.abs(self._last_action)) / self.act_dimension
	
			return state_reward + action_reward
	models/TD_121120_2336
		Changed QFBEnv.objective:
		def objective(self):
			state_reward = -np.sqrt(np.mean(np.power(self._current_state, 2)))
			action_reward = -np.sqrt(np.mean(np.power(self._last_action, 2))) / 5
	
			return state_reward + action_reward
	
	models/TD_121220_0014
		Changed QFBEnv.objective back to previous configuration
		
	model/TD_121220_09
		Layers = [32, 32]
	'''
	model = TD3(TD3Policy, env, gamma=0.99, learning_rate=1e-4, buffer_size=50000, learning_starts=100,
				train_freq=100, gradient_steps=100, batch_size=1024, tau=0.005, policy_delay=2,
				action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5, random_exploration=0.0,
				verbose=1, tensorboard_log=None,  #'logs'
				_init_setup_model=True, policy_kwargs={'layers': [160, 80, 10]},
				full_tensorboard_log=False, seed=0, n_cpu_tf_sess=None)

	nb_steps = 50000
	MyReplayBuffer.setup_storage(size=nb_steps, n_obs=env.obs_dimension, n_act=env.act_dimension)

	model.learn(total_timesteps=nb_steps, log_interval=300, replay_wrapper=MyReplayBuffer,
				tb_log_name=model_name)

	MyReplayBuffer.save_storage('buffer_storage_tmp.pkl')

	save_path = os.path.join('models', model_name)
	model.save(save_path)
	print(f'Model saved to: {save_path}')