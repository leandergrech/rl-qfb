import numpy as np

from qfb_env import QFBEnv
from bbq.utils import magNormResponse, fitTuneMedianLessParams, noiseHarmonics
from bbq import settings as bs


class QFBNLEnv(QFBEnv):
    """
    In this environment, each action is scaled by the current rate in each respective quadrupole. If an action value is
    larger than the allowed current rate (>1) then the entire action vector is scaled down accordingly to remain within
    the operational boundary of the slowest magnet.
    Large deviations of the tune between adjacent steps are also not permitted with a limit of 0.01*F_s ~ 112.5 Hz
    A PI controller is also implemented, where its response can be retured per step.
    """
    def __init__(self, noise_std=0.0, **kwargs):
        super(QFBNLEnv, self).__init__(noise_std, **kwargs)
        # PI controller parameters
        PID_K0 = 1.0
        PID_TAU = 0.1 / (2 * np.pi)
        PID_ALPHA = 5 / (2 * np.pi)
        self.PID_Kp = (PID_K0 * PID_TAU) / PID_ALPHA
        self.PID_Ki = (PID_K0 / PID_ALPHA) * QFBNLEnv.T_s
        self.freqs = np.arange(2400, 2400 + bs.delta_f * 100 + 1, bs.delta_f)
        self.harmonic_freqs = np.arange(self.freqs[0], self.freqs[-1], 50.0)
        self.omegas = self.freqs * 2 * np.pi

        self.state_tm1 = np.zeros(self.obs_dimension)  # Store t-1 state for PI controller
        self.state_tm2 = np.zeros(self.obs_dimension)  # Store t-2 state for PI controller

    def perturb_state(self, dq):
        """

        :param dq:
        :return:
        """
        '''Convert dq state from range [-1,1] to Hz'''
        dq *= self.Q_LIMIT_HZ

        '''Get a random resonance and zeta for simulation'''
        q = np.random.uniform(self.freqs[10], self.freqs[-10])
        zeta = np.power(10, np.random.uniform(-2.5, -1.8))

        '''Simulate spectrum with resonance q + dq and zeta'''
        omega_res = (q + dq ) * np.sqrt(1 - np.square(zeta)) * 2 *  np.pi
        mag = magNormResponse(self.omegas, omega_res, zeta)
        mag += noiseHarmonics(self.freqs, self.harmonic_freqs, 0.5, 2)
        bq_q = fitTuneMedianLessParams(mag, self.freqs)

        '''Get perturbed state and convert to range [-1,1]'''
        dq_perturbed = (bq_q - q) / self.Q_LIMIT_HZ

        return dq_perturbed

    def step(self, action):
        """
        :param action:
        :return:
        """
        '''Rate limit actions'''
        max_abs_action = max(abs(action))
        if max_abs_action > 1:
            temp_action = np.divide(action, max_abs_action)
        else:
            temp_action = action.copy()

        '''Calculate delta current from normalised action passed'''
        action_currents = np.array([a * self.calibrations[circ]['Irate'] * self.T_s for a, circ in zip(temp_action, self.circuits)])

        '''Get effective tune shift
            Convert from [-Q_LIMIT_HZ,Q_LIMIT_HZ] -> [-1,1]'''
        trim_tunes = self.rm.dot(action_currents)
        trim_tunes /= self.Q_LIMIT_HZ

        '''Cycle history and update current state'''
        self.state_tm2 = self.state_tm1
        self.state_tm1 = self.current_state
        self.current_state += trim_tunes

        '''Simulate state perturbation due to effect of 50 Hz harmonics'''
        self.current_state = np.array([self.perturb_state(o) for o in self.current_state])

        self.reward = self.objective(self.current_state)
        done = self.is_done()

        return self.current_state, self.reward, done, {}

    def pi_controller_action(self):
        """

        :return:
        """
        '''Convert from [-1,1] to [-Q_LIMIT_HZ,Q_LIMIT_HZ]'''
        current_state = self.current_state * self.Q_LIMIT_HZ
        state_tm1 = self.state_tm1 * self.Q_LIMIT_HZ

        '''Clip to maximum state rate allowed'''
        current_state = np.clip(a=current_state, a_min=-self.Q_STEP_MAX_HZ, a_max=self.Q_STEP_MAX_HZ)
        state_tm1 = np.clip(a=state_tm1, a_min=-self.Q_STEP_MAX_HZ, a_max=self.Q_STEP_MAX_HZ)

        '''PI controller velocity form'''
        pi_state = self.PID_Kp * (current_state - state_tm1) + self.PID_Ki * state_tm1
        pi_action = self.pi.dot(-pi_state)

        '''Normalize wrt maximum current rate per circuit'''
        for i, circ in enumerate(self.circuits):
            pi_action[i] = pi_action[i] / (self.calibrations[circ]['Irate'] * self.T_s)

        max_pi_rate = np.clip(a=np.max(np.abs(pi_action)), a_min=1.0, a_max=None)
        pi_action /= max_pi_rate

        return pi_action

def comparing_qfbenv_qfbnlenv():
    import matplotlib.pyplot as plt

    observations1 = np.zeros(shape=(0, 2))
    observations2 = np.zeros(shape=(0, 2))
    env1 = QFBNLEnv()
    env2 = QFBEnv()
    o1 = o2 = env1.reset()
    env2.reset(o2)

    observations1 = np.vstack([observations1, o1])
    observations2 = np.vstack([observations2, o2])

    for _ in range(100):
        a1 = env1.pi_controller_action()
        o1 = env1.step(a1)[0]
        observations1 = np.vstack([observations1, o1])

        a2 = env2.get_optimal_action(o2)
        o2 = env2.step(a2)[0]
        observations2 = np.vstack([observations2, o2])

    observations1 = observations1.T * env1.Q_LIMIT_HZ
    observations2 = observations2.T * env1.Q_LIMIT_HZ

    fig, ax = plt.subplots()
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

if __name__ == '__main__':
    comparing_qfbenv_qfbnlenv_noisy_tunes()











