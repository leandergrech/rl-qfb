from stable_baselines import TD3
from stable_baselines.td3.policies import FeedForwardPolicy as TD3Policy

model = TD3(CustomTD3Policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=buffer_size, learning_starts=100,
				train_freq=100, gradient_steps=100, batch_size=batch_size, tau=0.005, policy_delay=2,
				action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5, random_exploration=0.0,
				verbose=1, tensorboard_log=tb_log_loc, _init_setup_model=True, policy_kwargs={'layers':layers},
				full_tensorboard_log=False, seed=0, n_cpu_tf_sess=None)