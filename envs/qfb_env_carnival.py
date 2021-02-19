import numpy as np
from .qfb_env import QFBEnv

"""
This environment behaves like QFBEnv with the different that now there is 
a chance that an action will become inactive (masked i.e. carnival) during 
a random time in the environment, and will remain masked until the 
environment is reset.
"""
class QFBEnvCarnival(QFBEnv):
    PROB_TO_MASK = 0.01
    def __init__(self, *args, **kwargs):
        super(QFBEnvCarnival, self).__init__(*args, **kwargs)
        self.action_mask = np.ones(self.act_dimension)

    def step(self, action, noise_std=0.1):
        if np.random.uniform(0, 1) < QFBEnvCarnival.PROB_TO_MASK:
            self.action_mask[np.random.choice(self.act_dimension)] = 0
        action *= self.action_mask
        return super(QFBEnvCarnival, self).step(action, noise_std)

    def reset(self, init_state=None):
        self.action_mask = np.ones(self.act_dimension)
        return super(QFBEnvCarnival, self).reset(init_state)
