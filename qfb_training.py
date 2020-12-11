import os

from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy as TD3Policy
from datetime import datetime as dt

from qfb_env import QFBEnv

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':
	model_name = f"TD_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"
	env = QFBEnv()

	model = TD3(TD3Policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000, learning_starts=100,
					train_freq=100, gradient_steps=100, batch_size=128, tau=0.005, policy_delay=2,
					action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5, random_exploration=0.0,
					verbose=1, tensorboard_log='logs', _init_setup_model=True, policy_kwargs={'layers':[100, 100]},
					full_tensorboard_log=False, seed=0, n_cpu_tf_sess=None)

	model.learn(total_timesteps=100000, log_interval=100, tb_log_name=model_name)

	save_path = os.path.join('models', model_name)
	model.save(save_path)
	print(f'Model saved to: {save_path}')