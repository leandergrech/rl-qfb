import os
import numpy as np
import matplotlib.pyplot as plt

# from qfb_env import QFBEnv
# from bbq.utils import magNormResponse, fitTuneMedianLessParams, noiseHarmonics
# from bbq import settings as bs
from .qfb_env import QFBEnv
from .bbq.utils import magNormResponse, fitTuneMedianLessParams, noiseHarmonics
from .bbq import settings as bs

class QFBNLEnv(QFBEnv):
    """
    In this environment, each action is scaled by the current rate in each respective quadrupole. If an action value is
    larger than the allowed current rate (>1) then the entire action vector is scaled down accordingly to remain within
    the operational boundary of the slowest magnet.
    Large deviations of the tune between adjacent steps are also not permitted with a limit of 0.01*F_s ~ 112.5 Hz
    A PI controller is also implemented, where its response can be retured per step.
    """
    def __init__(self, noise_std=0.0, perturb_state=True, **kwargs):
        super(QFBNLEnv, self).__init__(noise_std=noise_std, **kwargs)
        # PI controller parameters
        # PID_K0 = 1.0
        # PID_TAU = (1/12.5) / (2 * np.pi) ## 0.1 / 2pi # characteristic circuit time constant
        # PID_ALPHA = (0.1/12.5) / (2 * np.pi) ## 5 / (2pi) # desired loop time constant
        # self.PID_Kp = ((PID_K0 * PID_TAU) / PID_ALPHA)
        # self.PID_Ki = ((PID_K0 / PID_ALPHA) * QFBNLEnv.T_s)
        self.PID_Kp = 1000.0
        self.PID_Ki = 2000.0
        self.MAX_TUNE_CHANGE_RATE = 0.01 # [(tune fractional part)/second]
        self.MAX_TUNE_CHANGE_Ts = self.MAX_TUNE_CHANGE_RATE * self.T_s

        self.max_steps = self.EPISODE_LENGTH_LIMIT

        # Perturbation parameters
        self._perturb = perturb_state
        self.init_freq = init_freq = 3300
        self._freqs = np.arange(init_freq, init_freq + bs.delta_f * 100 + 1, bs.delta_f)
        self._omegas = self._freqs * 2 * np.pi
        self._harmonic_freqs = np.arange(self._freqs[0], self._freqs[-1], 50.0)
        self.perturbation_center_freq = 0.3133 * self.F_s

        self.current_state_tm0 = np.zeros(self.obs_dimension)  # Store t-0 state for PI controller
        self.current_state_tm1 = np.zeros(self.obs_dimension)  # Store t-1 state for PI controller
        self.current_state_tm2 = np.zeros(self.obs_dimension)  # Store t-2 state for PI controller

    def reset(self, init_state=None):
        o = super(QFBNLEnv, self).reset(init_state)

        clipped_o = np.clip(o, -self.MAX_TUNE_CHANGE_Ts, self.MAX_TUNE_CHANGE_Ts)
        np.copyto(self.current_state_tm0, clipped_o)
        np.copyto(self.current_state_tm1, clipped_o)
        np.copyto(self.current_state_tm2, clipped_o)
        # self.current_state_tm1 = np.zeros(self.obs_dimension)
        # self.current_state_tm2 = np.zeros(self.obs_dimension)

        return o

    def __repr__(self):
        return 'QFBNLEnv'
        
    def perturb_state(self, dq):
        """

        :param dq: state to perturb
        :return:
        """
        '''Convert dq state from range [-1,1] to Hz'''
        dq *= self.Q_LIMIT_HZ

        '''Get a random resonance and zeta for simulation'''
        q = self.perturbation_center_freq
        zeta = np.power(10, np.random.uniform(-2.5, -1.8))

        '''Simulate spectrum with resonance q + dq and zeta'''
        omega_res = (q + dq ) * np.sqrt(1 - 2 * np.square(zeta)) * 2 *  np.pi
        mag = magNormResponse(self._omegas, omega_res, zeta)
        mag += np.random.normal(0.0, 0.1, 101)
        mag += noiseHarmonics(self._freqs, self._harmonic_freqs, 0.1, 1.1)
        bq_q = fitTuneMedianLessParams(mag, self._freqs)

        '''Get perturbed state and convert to range [-1,1]'''
        dq_perturbed = (bq_q - q) / self.Q_LIMIT_HZ

        return dq_perturbed

    def step(self, action):
        """
        :param action:
        :return:
        """
        temp_action = np.copy(action)
        temp_action += np.random.normal(0.0, self.noise_std, self.act_dimension)
        # '''Rate limit actions'''
        # max_abs_action = max(abs(temp_action))
        # if max_abs_action > 1:
        #     temp_action = np.divide(temp_action, max_abs_action)
        '''Rate limit actions - independent limits'''
        clip_abs_action_denominator = np.clip(np.abs(temp_action), 1.0, None)
        temp_action = np.divide(temp_action, clip_abs_action_denominator)

        '''Calculate delta current from normalised action passed'''
        action_currents = np.array([a * self.calibrations[circ]['Irate'] * self.T_s for a, circ in zip(temp_action, self.circuits)])

        '''Get effective tune shift
            Convert from [-Q_LIMIT_HZ,Q_LIMIT_HZ] -> [-1,1]'''
        trim_tunes = self.rm.dot(action_currents)
        trim_tunes /= self.Q_LIMIT_HZ

        '''Update current state'''
        # self.current_state_tm1 = deepcopy(self.current_state)
        self.current_state += trim_tunes

        '''Simulate state perturbation due to effect of 50 Hz harmonics'''
        if self._perturb:
            self.current_state = np.array([self.perturb_state(o) for o in self.current_state])

        '''Cycle history for PI controller'''
        self.pi_cycle_history(self.current_state)

        self.reward = self.objective(self.current_state)
        done, success = self.is_done()

        return self.current_state, self.reward, done, {'is_success': success}

    def objective(self, state):
        reward = -np.sum(np.square(state))/ self.obs_dimension
        return reward * self.REWARD_SCALE

    def pi_controller_warm_up(self):
        for i in np.arange(2, -1, -1):
            self.pi_cycle_history(self.current_state * np.power(2, i))

    def pi_cycle_history(self, state):
        np.copyto(self.current_state_tm2, self.current_state_tm1)
        np.copyto(self.current_state_tm1, self.current_state_tm0)
        np.copyto(self.current_state_tm0,
                  np.clip(state * (self.Q_LIMIT_HZ / self.F_s), -self.MAX_TUNE_CHANGE_Ts,
                          self.MAX_TUNE_CHANGE_Ts))

    def pi_controller_action(self):
        """
        :return:
        """
        '''Convert from [-1,1] to [-Q, Q]'''
        state_tm0 = np.copy(self.current_state_tm0)
        state_tm1 = np.copy(self.current_state_tm1)
        state_tm2 = np.copy(self.current_state_tm2)

        # '''Clip to maximum state rate allowed and convert to fractional part tune'''
        # current_state = np.clip(a=current_state, a_min=-self.Q_STEP_MAX_HZ, a_max=self.Q_STEP_MAX_HZ) / (self.F_s)
        # state_tm1 = np.clip(a=state_tm1, a_min=-self.Q_STEP_MAX_HZ, a_max=self.Q_STEP_MAX_HZ) / (self.F_s)
        # state_tm2 = np.clip(a=state_tm2, a_min=-self.Q_STEP_MAX_HZ, a_max=self.Q_STEP_MAX_HZ) / (self.F_s)

        '''Apply controller'''
        if self.it > 1 and self.PID_Ki > 0.0:
            '''PI controller velocity form'''
            pi_state = self.PID_Kp * (state_tm0 - state_tm1) + self.PID_Ki * state_tm1
        else:
            '''P controller'''
            pi_state = self.PID_Kp * state_tm0

        '''Calculate action from QPI'''
        pi_action = -self.pi.dot(pi_state)

        '''Normalize wrt maximum current rate per circuit'''
        for i, circ in enumerate(self.circuits):
            pi_action[i] = pi_action[i] / (self.calibrations[circ]['Irate'] * self.T_s)

        pi_action = np.clip(pi_action, -1.0, 1.0)

        return pi_action

def comparing_qfbenv_qfbnlenv():
    import matplotlib.pyplot as plt

    observations1 = np.zeros(shape=(0, 2))
    observations2 = np.zeros(shape=(0, 2))
    env1 = QFBNLEnv(perturb_state=False)
    env2 = QFBEnv()
    o1 = env1.reset()
    o2 = np.copy(o1)
    env2.reset(o2)

    observations1 = np.vstack([observations1, o1])
    observations2 = np.vstack([observations2, o2])

    d1 = False
    d2 = False
    for _ in range(100):
        a1 = env1.pi_controller_action()
        o1, _, d1, _ = env1.step(a1)
        observations1 = np.vstack([observations1, o1])

        a2 = env2.get_optimal_action(o2)
        o2, _, d2, _ = env2.step(a2)
        observations2 = np.vstack([observations2, o2])

        if d1 and d2:
            break

    observations1 = observations1.T * env1.Q_LIMIT_HZ
    observations2 = observations2.T * env1.Q_LIMIT_HZ

    fig, ax = plt.subplots()
    ax.axhline(env1.Q_GOAL_HZ, color='g', ls='--')
    ax.axhline(-env1.Q_GOAL_HZ, color='g', ls='--')
    ax.plot(observations1[0], 'b', label='PI - dQh')
    ax.plot(observations1[1], 'b', label='PI - dQv')
    ax.plot(observations2[0], 'r', label='Opt - dQh')
    ax.plot(observations2[1], 'r', label='Opt - dQv')

    ax.legend()

    plt.show()

def comparing_qfbenv_qfbnlenv_noisy_tunes():
    import matplotlib.pyplot as plt

    observations1 = np.zeros(shape=(0, 2))
    observations2 = np.zeros(shape=(0, 2))

    fig, ax = plt.subplots()
    ax.axhline(0, color='k', ls='dashed')

    env1 = QFBNLEnv()
    env2 = QFBNLEnv()
    o1 = o2 = np.array([10, -10], dtype=np.float)
    env1.reset(o1)
    env2.reset(o2)

    observations1 = np.vstack([observations1, o1])
    observations2 = np.vstack([observations2, o2])

    maxa1 = -np.inf
    maxa2 = -np.inf
    for _ in range(100):
        env1.current_state = o1
        env2.current_state = o2

        a1 = env1.pi_controller_action()
        o1 = env1.step(a1)[0]
        observations1 = np.vstack([observations1, o1])

        a2 = env2.get_optimal_action(o2)
        o2 = env2.step(a2)[0]
        observations2 = np.vstack([observations2, o2])

        maxa1 = max(maxa1, max(abs(a1)))
        maxa2 = max(maxa2, max(abs(a2)))

    print(maxa1, maxa2)

    observations1 = observations1.T * env1.Q_LIMIT_HZ
    observations2 = observations2.T * env1.Q_LIMIT_HZ

    qh1, = ax.plot(observations1[0], 'b', label='PI - dQh')
    qv1, = ax.plot(observations1[1], 'b', label='PI - dQv')
    qh2, = ax.plot(observations2[0], 'r', label='Opt - dQh')
    qv2, = ax.plot(observations2[1], 'r', label='Opt - dQv')

    ax.legend()

    plt.show()

def test_quad_malfunction():
    from collections import defaultdict
    import matplotlib.pyplot as plt

    envgood = QFBNLEnv(perturb_state=True)
    envbad = QFBNLEnv(perturb_state=True)
    init_o = np.array([0.5, 0.5])
    envgood.reset(init_o.copy())
    envbad.reset(init_o.copy())

    print(envbad.rm.dot(envbad.pi))
    print(envgood.rm.dot(envgood.pi))

    observations = dict(good = np.zeros(shape=(0, envgood.obs_dimension)),
             bad = np.zeros(shape=(0, envgood.obs_dimension)))
    actions = dict(good=np.zeros(shape=(0, envgood.act_dimension)),
             bad=np.zeros(shape=(0, envgood.act_dimension)))
    rewards = defaultdict(list)
    nb_steps = 100
    for i in range(nb_steps):
        for e, k in zip((envgood, envbad), ('good', 'bad')):
            a = e.pi_controller_action()
            if k in 'bad':
                a[2:6] = 0.0
            o, r, *_ = e.step(a)
            observations[k] = np.vstack([observations[k], o])
            actions[k] = np.vstack([actions[k], a])
            if i == 0:
                prev_cum = 0.0
            else:
                prev_cum = rewards[k][-1]
            rewards[k].append(r + prev_cum)

    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    axr = fig.add_subplot(gs[:, 2:])

    ax1.set_title('Good env')
    ax3.set_title('Bad env')

    ax1.set_title('Good - obs')
    ax1.plot(observations['good'])#, color='b')

    ax2.set_title('Good - act')
    ax2.plot(actions['good'])#, color='r')

    ax3.set_title('Bad - obs')
    ax3.plot(observations['bad'])#, color='b')

    ax4.set_title('Bad - act')
    ax4.plot(actions['bad'])#, color='r')

    axr.set_title('Cumulative rewards')
    axr.axhline(0.0, color='k', label='Convergence')
    axr.plot(rewards['good'], color='b', label='good env')
    axr.plot(rewards['bad'], color='r', label='bad env')
    axr.legend()

    plt.show()

def qfbnlenv_random_walk():
    env_kwargs = dict(rm_loc=os.path.join('..', 'metadata', 'LHC_TRM_B1.response'),
                      calibration_loc=os.path.join('..', 'metadata', 'LHC_circuit.calibration'),
                      perturb_state=False,
                      noise_std=0.1)
    env = QFBNLEnv(**env_kwargs)
    n_obs = env.obs_dimension
    n_act = env.act_dimension

    o = env.reset()
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.axhline(0.0, color='k', ls='--')
    ax1.axhspan(-env.Q_goal, env.Q_goal, color='g', alpha=0.4)
    ax2.axhline(0.0, color='k', ls='--')
    ax1.set_title('Observations')
    ax2.set_title('Actions')
    ax1.set_ylim((-1, 1))
    ax2.set_ylim((-1, 1))
    fig.tight_layout()

    plt.ion()
    obsline, = ax1.plot(range(n_obs), np.zeros(n_obs), color='b')
    actline, = ax2.step(range(n_act), np.zeros(n_act), color='r')
    rewards = []
    for _ in range(100):
        a = env.action_space.sample()
        # a = env.pi_controller_action()
        o, r, d, _ = env.step(a)
        rewards.append(r)

        obsline.set_ydata(o)
        actline.set_ydata(a)

        plt.pause(0.1)

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title('Rewards')

    plt.show()

def average_optimal_episode_length():
    from tqdm import tqdm
    env_kwargs = dict(rm_loc=os.path.join('..', 'metadata', 'LHC_TRM_B1.response'),
                      calibration_loc=os.path.join('..', 'metadata', 'LHC_circuit.calibration'),
                      perturb_state=False,
                      noise_std=0.1)
    env = QFBNLEnv(**env_kwargs)

    print(f'Reward success threshold: {env.objective([QFBEnv.Q_goal] * 2)}')

    o = env.reset()

    solved_len = []
    for ep in tqdm(range(1000)):
        for i in range(10000):
            a = env.pi_controller_action()
            o, r, d, _ = env.step(a)

            if d:
                solved_len.append(i)
                break
        o = env.reset()
    mu_unicode = '\u03bc'
    sigma_unicode = '\u03c3'
    print(f'Average episode length: {mu_unicode} = {np.mean(solved_len)}\t{sigma_unicode} = {np.std(solved_len)}')

def tuning_pi_controller():
    env_kwargs = dict(rm_loc=os.path.join('..', '..', 'metadata', 'LHC_TRM_B1.response'),
                      calibration_loc=os.path.join('..', '..', 'metadata', 'LHC_circuit.calibration'),
                      perturb_state=False,
                      noise_std=0.1)
    env = QFBNLEnv(**env_kwargs)
    env.PID_Kp = 1000
    env.PID_Ki = 2000
    print(f'PID_Kp = {env.PID_Kp}\tPID_Ki = {env.PID_Ki}')

    np.random.seed(123)

    o = env.reset()
    d = False
    a = np.zeros(env.act_dimension)

    env.pi_controller_warm_up()

    oss = [o.copy()]
    ass = [a]
    rss = [env.objective(o)]
    while not d:
    # for _ in range(1000):
        a = env.pi_controller_action()# + np.random.normal(0, 0.25, 16)
        o, r, d, _ = env.step(a)

        oss.append(o.copy())
        ass.append(a)
        rss.append(r)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(f'PID_Kp = {env.PID_Kp}  PID_Ki = {env.PID_Ki}')
    ax1.plot(np.array(oss).T[0], np.array(oss).T[1], marker='x', color='b')
    ax1.axhline(-env.Q_goal, color='g', ls='dashed')
    ax1.axhline(env.Q_goal, color='g', ls='dashed')
    ax1.axvline(-env.Q_goal, color='g', ls='dashed')
    ax1.axvline(env.Q_goal, color='g', ls='dashed')
    ax2.plot(np.array(ass), color='r')
    plt.show()









if __name__ == '__main__':
    # average_optimal_episode_length()
    tuning_pi_controller()










