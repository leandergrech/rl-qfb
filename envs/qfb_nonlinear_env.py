import numpy as np

from .qfb_env import QFBEnv


class QFBNLEnv(QFBEnv):
    # PI controller parameters
    PID_K0 = 1.0
    PID_TAU = 0.1 / (2 * np.pi)
    PID_ALPHA = 5 / (2 * np.pi)
    PID_Kp = (PID_K0 * PID_TAU) / PID_ALPHA
    T_s = 1 / 12.5
    PID_Ki = (PID_K0 / PID_ALPHA) * T_s

    def __init__(self, noise_std=0.0, **kwargs):
        QFBNLEnv.Q_INIT_STD_HZ = 100
        QFBNLEnv.Q_GOAL_HZ = 1
        QFBNLEnv.Q_LIMIT_HZ = 200
        QFBNLEnv.EPISODE_LENGTH_LIMIT = 40

        super(QFBNLEnv, self).__init__(noise_std, **kwargs)

        self.state_tm1 = np.zeros(self.obs_dimension)  # Store t-1 action for PI controller
        self.state_tm2 = np.zeros(self.obs_dimension)  # Store t-2 action for PI controller



    def step(self, action):
        """

        :param action:
        :return:
        """
        '''Rate limit actions'''
        max_abs_action = max(abs(action))
        if max_abs_action > 1:
            action /= max_abs_action

        '''Calculate delta current from noramlised action passed'''
        action_currents = np.zeros_like(action)
        for i, circ in enumerate(self.circuits):
            action_currents[i] = action[i] * self.calibrations[circ]['Irate'] * self.T_s

        trim_tunes = self.rm.dot(action_currents)
        # Convert from [0,0.5] -> [-1,1]
        trim_tunes = trim_tunes * 4.0 - 1.0

        self.state_tm2 = self.state_tm1
        self.state_tm1 = self.current_state
        self.current_state += trim_tunes

        self.reward = self.objective(self.current_state)
        done = self.is_done()
        self.it += 1

        return self.current_state, self.reward, done, {}

    def pi_controller_action(self):
        """"""
        '''PI controller velocity form'''
        pi_state = QFBNLEnv.PID_Kp * (self.current_state - self.state_tm1) + QFBNLEnv.PID_Ki * self.state_tm1
        pi_action = -self.pi.dot(pi_state)

        '''Normalize wrt maximum current rate per circuit'''
        for i, circ in enumerate(self.circuits):
            pi_action[i] = pi_action[i] / (self.calibrations[circ]['Irate'] * self.T_s)

        max_pi_rate = np.max(np.abs(pi_action))
        if max_pi_rate > 1:
            pi_action /= max_pi_rate

        return pi_action

if __name__ == '__main__':
    env = QFBNLEnv()
    o = env.reset()

    for _ in range(10):
        a = env.pi_controller_action()
        env.step(a)[0]











