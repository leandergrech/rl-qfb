import gym
from gym.spaces import Box
import numpy as np

class QFBEnv(gym.Env):
    rm_loc = 'LHC_TRM_B1.response'
    F_s = 11245.55
    Q_init_std_hz = 50
    Q_goal_hz = 1
    Q_limit_hz = 100
    obs_norm = Q_limit_hz / F_s
    act_norm = 1e-2

    reward_accumulated_limit = -10

    def __init__(self):
        self._last_action = None
        self._current_state = None
        self._prev_state = None
        self._reward = None
        self._cum_reward = None

        self.__full_rm = None
        self.__knobNames = None
        self.__knobNormal = None
        self.__circuits = None
        self.rm = None
        self.quad_names = None

        self.__get_rm()
        self.__adjust_qrm()

        self.obs_dimension = self.rm.shape[1]
        self.act_dimension = self.rm.shape[0]

        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.obs_dimension,), dtype=np.float)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.act_dimension,), dtype=np.float)

    def reset(self):
        self._current_state = np.random.normal(0, self.Q_init_std_hz / (self.F_s * self.obs_norm), 2)
        self._prev_state = self._current_state
        self._last_action = np.zeros(self.act_dimension)
        self._cum_reward = 0

        return self._current_state

    def step(self, action):
        self._last_action = action

        # Convert action to rm units and calculate real trim
        trim_state = self.rm.T.dot(np.multiply(action, self.act_norm))
        # Normalise trim obtained from action
        trim_state = np.divide(trim_state, self.obs_norm)

        self._prev_state = self._current_state
        self._current_state = self._current_state - trim_state

        self._reward = self.objective()
        self._cum_reward += self._reward
        done = self.is_done()

        return self._current_state, self._reward, done, {}

    def objective(self):
        return -np.sum(np.abs(self._current_state))

    def is_done(self):
        ave_rew = np.abs(self._reward) / 2

        if ave_rew < self.Q_goal_hz / (self.F_s * self.obs_norm) or ave_rew > self.Q_limit_hz / (self.F_s * self.obs_norm):
            done = True
        elif self._cum_reward < self.reward_accumulated_limit:
            done = True
        else:
            done = False

        return done

    def get_state(self):
        return self._current_state

    def get_state_prev(self):
        return self._prev_state

    def get_action_prev(self):
        return self._last_action

    def get_reward_cum(self):
        return self._cum_reward

    def __get_rm(self, reload=False):
        if self.__full_rm is not None and not reload:
            return self.__full_rm

        rm = []
        circuits = []
        reading_matrix = False
        reading_circuits = False

        def split_line_ret_floats(line):
            return [float(item) for item in line.split()]

        with open(self.rm_loc, 'r') as f:
            contents = f.readlines()
            for line in contents:
                if '# matrix' in line:
                    reading_matrix = True
                    reading_circuits = False
                    continue
                elif '# knobNames' in line:
                    line = line.split(':')[1]
                    self.__knobNames = [item.split('/')[1].lower() for item in line.split()]
                    continue
                elif '# knobNormal' in line:
                    line = line.split(':')[1]
                    self.__knobNormal = split_line_ret_floats(line)
                    continue
                elif '# circuits' in line:
                    reading_circuits = True
                    reading_matrix = False
                    continue

                if reading_matrix:
                    if '#' in line:
                        reading_matrix = False
                        continue
                    else:
                        rm.append(split_line_ret_floats(line))
                elif reading_circuits:
                    if '#' in line:
                        reading_circuits = False
                        continue
                    else:
                        circuits.append(line.lower())

        self.__full_rm = np.array(rm)
        self.__circuits = np.array(circuits)

        return self.__full_rm

    def __adjust_qrm(self):
        assert self.__full_rm is not None, "__get_rm() was not called, RM uninitialised"

        trunc_rm = self.__full_rm[:, :2]

        good_quads = []
        good_idx = []
        for i, (quad, rm_vals) in enumerate(zip(self.__circuits, trunc_rm)):
            if np.sum(np.abs(rm_vals)) > 0.0:
                good_quads.append(quad)
                good_idx.append(i)

        self.quad_names = good_quads
        self.rm = self.__full_rm[good_idx, :2]


if __name__ == '__main__':
    env = QFBEnv()

    o = env.reset()
    n_act = env.act_dimension
    n_obs = env.obs_dimension

    nb = 100

    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2)
    l, = ax.plot(o)
    ax.axhline(env.Q_goal_hz / (env.F_s * env.obs_norm), color='g', ls='dashed')
    ax.axhline(env.Q_limit_hz / (env.F_s * env.obs_norm), color='r', ls='dashed')
    ax.axhline(-env.Q_goal_hz / (env.F_s * env.obs_norm), color='g', ls='dashed')
    ax.axhline(-env.Q_limit_hz / (env.F_s * env.obs_norm), color='r', ls='dashed')

    cr_line, = ax2.plot([], [])
    ax2.set_xlim((0, nb))
    ax2.axhline(env.reward_accumulated_limit)

    plt.ion()

    crs = []

    for i in range(nb):
        action = env.action_space.sample()
        print(action)
        o, r, d, _ = env.step(action)
        crs.append(env.get_reward_cum())

        l.set_ydata(o)
        cr_line.set_data(range(i + 1), crs)
        ax2.set_ylim((crs[-1], 0))
        print(d)



        plt.pause(0.1)





