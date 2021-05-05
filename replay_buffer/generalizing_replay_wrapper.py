import numpy as np

from stable_baselines.common.buffers import ReplayBuffer


class GeneralizingReplayWrapper(object):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

        self.generalising_factor = 0.2 # Defined as the percentage of data which is coming from real experience
        self.noise_std = 0.05
        # self._storage = []
        # self._maxsize = size
        # self._next_idx = 0

    def can_sample(self, *args, **kwargs):
        return self.replay_buffer.can_sample(*args, **kwargs)

    def add(self, *args, **kwargs):
        return self.replay_buffer.add(*args, **kwargs)

    def extend(self, *args, **kwargs):
        return self.replay_buffer.extend(*args, **kwargs)

    def sample(self, batch_size, env, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        nb_traj_samples = int(batch_size * self.generalising_factor)
        traj_data = self.replay_buffer.sample(batch_size=nb_traj_samples)

        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in range(batch_size):
            j = i % nb_traj_samples
            obs_t, act, rew, obs_tp1, done = \
                traj_data[0][j], traj_data[1][j], traj_data[2][j], traj_data[3][j], traj_data[4][j]

            obs_t += np.random.normal(0.0, self.noise_std, len(obs_t))
            obs_tp1 += np.random.normal(0.0, self.noise_std, len(obs_t))
            act += np.random.normal(0.0, self.noise_std, len(act))

            obses_t.append(obs_t)
            actions.append(act)
            rewards.append(rew)
            obses_tp1.append(obs_tp1)
            dones.append(done)

        return (np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones))
