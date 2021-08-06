import os

from stable_baselines import TD3, HER
from datetime import datetime as dt

from qfb_env.qfb_env import QFBGoalEnv

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':
	model_name = f"HER_TD_{dt.strftime(dt.now(), '%m%d%y_%H%M')}"
	env = QFBGoalEnv()

	model = HER('MlpPolicy', env, TD3, n_sampled_goal=4, goal_selection_strategy='future', tensorboard_log='logs', policy_kwargs={'layers': [32, 32]})

	model.learn(total_timesteps=300000, log_interval=300, tb_log_name=model_name)
	save_path = os.path.join('models', model_name)
	model.save(save_path)
	print(f'Model saved to: {save_path}')