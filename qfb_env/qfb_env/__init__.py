from gym.envs.registration import register

register(id="qfbenv-v1", entry_point="qfb_env.envs:QFBEnv")
register(id="qfbnlenv-v1", entry_point="qfb_env.envs:QFBNLEnv")