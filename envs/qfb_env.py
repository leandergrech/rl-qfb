import os
from collections import defaultdict, deque
import gym
from gym.spaces import Box
import numpy as np


class QFBEnv(gym.Env):
    rm_loc = os.path.join('../metadata', 'LHC_TRM_B1.response')
    calibration_loc = os.path.join('../metadata', 'LHC_circuit.calibration')
    F_s = 11245.55
    Q_init_std_hz = 50

    Q_goal_hz = 1
    Q_limit_hz = 100

    Q_init_std = Q_init_std_hz / Q_limit_hz
    Q_goal = Q_goal_hz / Q_limit_hz

    act_norm = 200
    obs_lim = 5

    # reward_accumulated_limit = -10
    episode_length_limit = 40

    def __init__(self, noise_std=None):
        self.noise_std = noise_std
        self._last_action = None
        self._current_state = None
        self._prev_state = None
        self._reward = None
        self._reward_thresh = self.objective([QFBEnv.Q_goal]*2)
        self._reward_deque = deque(maxlen=5)

        self.__full_rm = None
        self.__knobNames = None
        self.__knobNormal = None
        self.__circuits = None
        self.__calibrations = None
        self.rm = None
        self.pi = None
        self.quad_names = None

        self.__get_rm()
        self.__adjust_qrm()
        self.__load_circuit_calibrations()

        self.obs_dimension = self.rm.shape[1]
        self.act_dimension = self.rm.shape[0]

        self.observation_space = Box(low=-np.ones(self.obs_dimension), high=np.ones(self.obs_dimension), dtype=np.float)
                                     # , shape=(self.obs_dimension,), dtype=np.float)
        self.action_space = Box(low=-np.ones(self.act_dimension), high=np.ones(self.act_dimension), dtype=np.float)
                                # , shape=(self.act_dimension,), dtype=np.float)

        self.it = 0

    def reset(self, init_state=None):
        o = self._reset(init_state)
        return o

    def _reset(self, init_state=None):
        if init_state is None:
            # self._current_state = np.random.normal(0, self.Q_init_std, self.obs_dimension)
            self._current_state = self.observation_space.sample()
        else:
            self._current_state = init_state
        self._prev_state = self._current_state
        self._last_action = np.zeros(self.act_dimension)
        self.it = 0

        return self._current_state

    def step(self, action, noise_std=0.1):
        if self.noise_std is not None:
            noise_std = self.noise_std

        self._last_action = action


        # Convert action to rm units and add noise
        action += np.random.normal(scale=noise_std, size=self.act_dimension)
        action_denorm = np.multiply(action, self.act_norm)

        # Calculate real trim
        trim_state = self.rm.T.dot(action_denorm)
        # Normalise trim obtained from action
        trim_state = np.divide(trim_state, self.Q_limit_hz)

        self._prev_state = self._current_state
        self._current_state = self._current_state + trim_state

        self._reward = self.objective(self._current_state)
        self._reward_deque.append(self._reward)

        done = self.is_done()

        self.it += 1

        return self._current_state, self._reward, done, {}

    def objective(self, state):
        # state_reward = -np.sqrt(np.mean(np.power(self._current_state, 2)))
        # action_reward = -np.sqrt(np.mean(np.power(self._last_action, 2))) / 5
        state_reward = -np.square(np.sum(np.abs(state)) + 1)
        # for s in state:
        #     if np.abs(s) > 1:
        #         state_reward -= np.abs(s)
        # action_reward = -np.sum(np.abs(self._last_action)) / self.act_dimension

        return 1*state_reward #+ action_reward

    def is_done(self):
        # Reach goal
        if np.mean(self._reward_deque) > self._reward_thresh or \
                np.max(np.abs(self._current_state)) > QFBEnv.obs_lim:
            done = True
        else:
            done = False

        # Reach maximum time limit
        # if self.it < QFBEnv.episode_length_limit:
        #     done = False
        # else:
        #     done = True

        return done

    def get_optimal_action(self, state):
        state = np.multiply(state, self.Q_limit_hz)
        action_optimal = -np.divide(self.pi.T.dot(state), self.act_norm)
        action_optimal = np.clip(action_optimal, -1, 1)

        return action_optimal

    def get_state(self):
        return self._current_state

    def get_state_prev(self):
        return self._prev_state

    def get_action_prev(self):
        return self._last_action

    def get_knob_names(self):
        assert self.__knobNames is not None, "Environment has not been reset"
        return self.__knobNames

    def get_circuits(self):
        assert self.__circuits is not None, "Environment has not been reset"
        return self.__circuits

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
                        circuits.append(line.lower().split('/')[0])

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
        self.pi = np.linalg.pinv(self.rm)

    def __load_circuit_calibrations(self):
        """
        Load circuit calibration data. Used to convert uRads<-->Amperes
        :param circuit_calibration_file:
        :return: {device_name:{"circuit_name":..., "bnom":..., etc.}}
        """
        calib_file = os.path.abspath(self.calibration_loc)

        self._calibrations = defaultdict(dict)

        with open(calib_file) as f:
            keywords = None
            # Read and parse file
            for line in f:
                if '#' in line:
                    # KEYWORDS --> Occurrence of # means the keywords are on this line
                    keywords = [k for k in line.split() if ('|' not in k) and ('#' not in k)][1:]
                    parser = {}
                    for k in keywords:
                        if 'name' in k:
                            parser[k] = lambda x: x.strip('"').lower()
                        elif any(integer_col_hint in k for integer_col_hint in ["beam", "IR", "FGC", "idnbr"]):
                            parser[k] = lambda x: int(x)
                        else:
                            parser[k] = lambda x: float(x)
                else:
                    if keywords is None:
                        raise Exception("Keywords were not found at the top of file.")

                    split_line = np.array([item for item in line.split() if '|' not in item])
                    cod_name = split_line[1].lower()
                    split_line = split_line[1:]

                    for i, (k, item) in enumerate(zip(keywords, split_line)):
                        self._calibrations[cod_name][k] = parser[k](item)

    def get_calibrations(self):
        return self._calibrations


class QFBGoalEnv(QFBEnv):
    def __init__(self):
        super(QFBGoalEnv, self).__init__()
        self.observation_space = gym.spaces.Dict({
            'observation': Box(low=-1.0, high=1.0, shape=(self.obs_dimension,), dtype=np.float),
            'desired_goal': Box(low=-1.0, high=1.0, shape=(self.obs_dimension,), dtype=np.float),
            'achieved_goal': Box(low=-1.0, high=1.0, shape=(self.obs_dimension,), dtype=np.float)
        })
        self.desired_goal = np.zeros(self.obs_dimension)

    def reset(self):
        init_obs = super(QFBGoalEnv, self).reset()

        return {'observation': init_obs,
                'achieved_goal': init_obs,
                'desired_goal': self.desired_goal}

    def compute_reward(self, achieved_goal, desired_goal, info):
        difference_objective = self.objective(achieved_goal - desired_goal)

        if np.abs(difference_objective) <= self.Q_goal_hz / (self.F_s * self.obs_norm):
            return 0.0
        else:
            return -1.0

    def step(self, action, noise_std=0.05):
        o, r, d, info = super(QFBGoalEnv, self).step(action, noise_std)
        o_dict = {'observation': o,
                  'achieved_goal': o.copy(),
                  'desired_goal': self.desired_goal}

        return o_dict, r, d, info


def test_actions():
    import matplotlib.pyplot as plt
    plt.ion()

    env = QFBEnv()
    o = env.reset()

    fig, (ax1, ax2, ax3) = plt.subplots(3, num=1, gridspec_kw={'height_ratios':[2, 2, 1]})
    state_line, = ax1.plot(o)
    action_line, = ax2.plot(np.zeros(env.act_dimension))
    rew_line, = ax3.plot([], [])
    ax1.axhline(1.0, color='k', ls='dashed')
    ax1.axhline(env.Q_goal, color='g', ls='dashed')
    ax1.axhline(-1.0, color='k', ls='dashed')
    ax1.axhline(-env.Q_goal, color='g', ls='dashed')
    ax2.set_ylim((-1.1, 1.1))
    ax3.axhline(-env.Q_goal, color='g', ls='dashed')
    plt.pause(1)
    a_max = a_min = 0.0

    rewards = []
    for i in range(1000):

        # if i % 2 == 0:
        # a = env.get_optimal_action(o)
        # else:
        a = env.action_space.sample()

        a_max = np.max(np.concatenate([a, [a_max]]))
        a_min = np.min(np.concatenate([a, [a_min]]))
        o, r, d, _ = env.step(a, noise_std=0.0)
        rewards.append(r)
        state_line.set_ydata(o)
        action_line.set_ydata(a)
        rew_line.set_data(range(i + 1), rewards)

        ax2.set_ylim((a_min, a_max))
        ax3.set_ylim((min(rewards), 0.1))
        ax3.set_xlim((0, i+1))

        if d:
            plt.ioff()
            plt.show()
            print(f'Solved in {i} steps')
            print(f'Observation: {o}')
            break

        if plt.fignum_exists(1):
            plt.draw()
            plt.pause(0.1)
        else:
            break
    print(a_max, a_min)

def average_optimal_episode_length():
    from tqdm import tqdm
    env = QFBEnv()
    o = env.reset()

    solved_len = []
    for ep in tqdm(range(10000)):
        for i in range(10000):
            if i % 2 == 0:
                a = env.get_optimal_action(o)
            else:
                a = env.action_space.sample()
            o, r, d, _ = env.step(a)

            if d:
                solved_len.append(i)
                break
        o = env.reset()
    print(f'Average episode length: {np.mean(solved_len)}')

def average_optimal_episode_reward():
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    env = QFBEnv()
    o = env.reset()

    ep_rew = []
    for ep in tqdm(range(1000)):
        ep_rew.append(0.0)
        for i in range(10000):
            a = env.get_optimal_action(o)
            o, r, d, _ = env.step(a, noise_std=0.0)

            ep_rew[-1] += r

            if d:
                break
        o = env.reset()

    print(f'Optimal average episode reward: {np.mean(ep_rew)}')

    fig, ax = plt.subplots()
    ax.plot(ep_rew)
    ax.set_ylabel('Total reward')
    ax.set_xlabel('Episode')
    ax.set_title(f'Average episode reward: {np.mean(ep_rew)}')
    plt.show()



if __name__ == '__main__':
    test_actions()







