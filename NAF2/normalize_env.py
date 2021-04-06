import gym
import numpy as np


class NormalizeEnv(gym.Wrapper):
    '''
    Gym Wrapper to normalize the environment
    '''
    OBS_LOW = ACT_LOW = -1.0
    OBS_HIGH = ACT_HIGH = 1.0
    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)

        self.obs_dim = self.env.observation_space.shape
        self.obs_high = self.env.observation_space.high
        self.obs_low = self.env.observation_space.low
        self.act_dim = self.env.action_space.shape
        self.act_high = self.env.action_space.high
        self.act_low = self.env.action_space.low

        # state space definition
        self.observation_space = gym.spaces.Box(low=NormalizeEnv.OBS_LOW,
                                                high=NormalizeEnv.OBS_HIGH,
                                                shape=self.obs_dim,
                                                dtype=np.float64)

        # action space definition
        self.action_space = gym.spaces.Box(low=NormalizeEnv.ACT_LOW,
                                           high=NormalizeEnv.ACT_HIGH,
                                           shape=self.act_dim,
                                           dtype=np.float64)

    def reset(self, *args, **kwargs):
        return self.state_from_env(self.env.reset(*args, **kwargs))

    def step(self, action):
        # TODO: check the dimensions
        ob, reward, done, info = self.env.step(self.action_to_env(action))

        return self.state_from_env(ob), reward, done, info

    def action_to_env(self, act):
        scale = (self.act_high - self.act_low)
        return_value = (scale * act + self.act_high + self.act_low) / 2
        return return_value

    def state_from_env(self, ob):
        scale = (self.obs_high - self.obs_low)
        return (2 * ob - (self.obs_high + self.obs_low)) / scale


if __name__ == '__main__':
    from random_env import RandomEnv
    env = RandomEnv(5, 5, np.load('random_env_rms\\random_env_5x5.npy'))
    nenv = NormalizeEnv(env)

    print(nenv.state_from_env(env.observation_space.low))