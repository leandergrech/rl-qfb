import numpy as np
import pickle as pkl
from datetime import datetime as dt
from stable_baselines.common.buffers import ReplayBuffer


class MyReplayBuffer(ReplayBuffer):
    obs_t_storage = None
    action_storage = None
    reward_storage = None
    obs_tp1_storage = None
    done_storage = None

    episode_snips = []

    def __init__(self, size, obs_dim, act_dim):
        self.size = size
        self.replay = ReplayBuffer(size)
        self.i = 0

        self.prev_done = True # Initialised to True in order to catch the first episode (see add(..) method)

        self.cur_ep_start_idx = 0
        self.episode_performance = []

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.setup_storage()

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.replay.add(obs_t, action, reward, obs_tp1, done)
        MyReplayBuffer.obs_t_storage[self.i] = obs_t
        MyReplayBuffer.action_storage[self.i] = action
        MyReplayBuffer.reward_storage[self.i] = reward
        MyReplayBuffer.obs_tp1_storage[self.i] = obs_tp1
        MyReplayBuffer.done_storage[self.i] = done

        self.i += 1

        if self.prev_done is True and done is False:
            MyReplayBuffer.episode_snips.append((self.cur_ep_start_idx, self.i))
            self.cur_ep_start_idx = self.i

        self.prev_done = done

    def can_sample(self, n_samples):
        return self.replay.can_sample(n_samples)

    def sample(self, batch_size, *args, **_kwargs):
        return self.replay.sample(batch_size, *args, **_kwargs)

    # @staticmethod
    # Has to be static because I pass the wrapper to the training method not an instance of the buffer so storage has
    # to be allocated before
    def setup_storage(self):
        MyReplayBuffer.obs_t_storage = np.empty((self.size, self.obs_dim))
        MyReplayBuffer.action_storage = np.empty((self.size, self.act_dim))
        MyReplayBuffer.reward_storage = np.empty(self.size)
        MyReplayBuffer.obs_tp1_storage = np.empty((self.size, self.obs_dim))
        MyReplayBuffer.done_storage = np.empty(self.size)

    @staticmethod
    def save_storage(savepath=None):
        if savepath is None:
            savepath = f"MyReplayBuffer_{dt.strftime(dt.now(), '%m%d%y_%H%M')}.pkl"

        with open(savepath, 'wb') as f:
            dump_dict = {'obs_t': MyReplayBuffer.obs_t_storage,
                         'action': MyReplayBuffer.action_storage,
                         'reward': MyReplayBuffer.reward_storage,
                         'obs_tp1': MyReplayBuffer.obs_tp1_storage,
                         'done': MyReplayBuffer.done_storage,
                         'episode_snips': MyReplayBuffer.episode_snips}
            pkl.dump(dump_dict, f)


def create_random_trajectories():
    from envs.qfb_env import QFBEnv

    nb_steps = 1000

    env = QFBEnv()
    buffer = ReplayBuffer(size=nb_steps)
    buffer = MyReplayBuffer(buffer)
    buffer.setup_storage(nb_steps, env.obs_dimension, env.act_dimension)

    o = env.reset()

    for i in range(nb_steps):
        a = env.action_space.sample()
        otp1, r, d, _ = env.step(a)

        print(d)
        buffer.add(o, a, r, d, otp1)
        o = otp1
        if d:
            print('resetting')
            env.reset()

    buffer.save_storage('testing_buffer.pkl')


def viewing_random_trajectories():
    import matplotlib.pyplot as plt

    with open('testing_buffer.pkl', 'rb') as f:
        data = pkl.load(f)

    print(data['done'])
    fig, ax = plt.subplots()

if __name__ == '__main__':
    create_random_trajectories()



