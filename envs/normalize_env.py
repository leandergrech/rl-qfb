import gym


class NormalizeEnv(gym.Wrapper):
    '''
    Gym Wrapper to normalize the environment
    '''

    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)

        self.obs_dim = self.env.observation_space.shape
        self.obs_high = self.env.observation_space.high
        self.obs_low = self.env.observation_space.high
        self.act_dim = self.env.action_space.shape
        self.act_high = self.env.action_space.high
        self.act_low = self.env.action_space.low

        # state space definition
        self.observation_space = gym.spaces.Box(low=-1.0,
                                                high=1.0,
                                                shape=self.obs_dim,
                                                dtype=np.float64)

        # action space definition
        self.action_space = gym.spaces.Box(low=-1.0,
                                           high=1.0,
                                           shape=self.act_dim,
                                           dtype=np.float64)

    def reset(self, *args, **kwargs):
        return self.scale_state_env(self.env.reset(*args, **kwargs))

    def step(self, action):
        # TODO: check the dimensions
        ob, reward, done, info = self.env.step(self.descale_action_env(action))

        return self.scale_state_env(ob), reward, done, info

    def descale_action_env(self, act):
        scale = (self.act_high - self.act_low)
        return_value = (scale * act + self.act_high + self.act_low) / 2
        return return_value

    def scale_state_env(self, ob):
        scale = (self.obs_high - self.obs_low)
        return (2 * ob - (self.obs_high + self.obs_low)) / scale
