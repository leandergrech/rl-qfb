import numpy as np
import gym
from gym.spaces import Box

class RandomEnv(gym.Env):
    MAX_TIME = 20
    goal = 0.1

    def __init__(self, n_obs, n_act, rm=None):
        self.MAX_TIME = max_time
        self.obs_dimension = n_obs
        self.act_dimension = n_act
        self._verbose = verbose
        self._action_scale = 1.0

        if rm is None:

            self.rm = np.random.uniform(-1.0, 1.0, (n_obs, n_act))
        else:
            self.rm = rm

        self.objective_epsilon = objective_epsilon

        self.pinv = np.linalg.pinv(self.rm)

        self._init_state = None
        self._current_state = None
        self._last_action = None
        self.done = False

        self.reward = None
        self._init_reward = None

        self._curr_step = -1
        self._curr_episode = -1

        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(n_act,), dtype=np.float)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_obs,), dtype=np.float)

        self.vis = None
        self.reward_history = []
        self.terminal_history = []

        self.noise_std = noise_std
        self._last_action = None
        self._current_state = None
        self._prev_state = None
        self._reward = None
        self._reward_thresh = self.objective([QFBEnv.Q_goal] * 2)
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


	def reset(self):
		self._curr_episode += 1
		self._curr_step = 0
		self.done = False

		init_action = self.action_space.sample() * 10
		init_state = self.rm.dot(init_action)
		self._init_state = self._current_state = init_state
		self._init_reward = self.objective(init_state)

		self._last_action = np.zeros(self.act_dimension)

		return self._init_state

	def customReset(self, init_state):
		self._curr_episode += 1
		self._curr_step = 0
		self.done = False

		self._init_state = self._current_state = init_state
		self._init_reward = self.objective(init_state)

		self._last_action = np.zeros(self.act_dimension)

	def takeAction(self, action):
		action *= self._action_scale
		deltas = self.rm.dot(action)

		# noise = self.observation_space.sample() * 0.0
		self._current_state += deltas #+ noise

		self._current_state = np.clip(self._current_state, self.observation_space.low, self.observation_space.high)

		self._last_action = action


	def step(self, action):
		self._curr_step += 1
		self.takeAction(action)

		reward = self.objective(self._current_state)
		self.reward = reward
		self.reward_history.append(reward)


		if self._curr_step >= self.MAX_TIME:
			self.done = True
			self.terminal_history.append((self._curr_episode * self.MAX_TIME) + self._curr_step)

		return self._current_state, self.reward, self.done, {}

	def objective(self, state):
		return -np.sqrt(np.mean(np.power(state, 2)))

	def seed(self, seed=None):
		np.random.seed(seed)
		seeding.np_random(seed)

		return [seed]

	def optimal_action(self, state, scale=0.2):
		return -self.pinv.dot(self._current_state) * scale

	def makeProblemHarder(self, factor=1.1):
		self.objective_epsilon /= factor

	def getObjectiveEpsilon(self):
		return self.objective_epsilon

	def render(self, mode='human', nb_axes=1, last=False):
		if self.vis is None:
			plt.ion()
			self.vis = {}

			fig, axes = plt.subplots(nb_axes)
			fig.tight_layout(pad=0.5)
			fig.subplots_adjust(left=0.1, top=0.9)

			lines = {}
			if nb_axes > 1:
				ax = axes[0]
				self.vis['other_axes'] = axes[1:]
			else:
				ax = axes

			lines['rewards'], = ax.plot(self.reward_history, 'b')

			_temp_line = plt.Line2D([], [], c='k', ls='dashed')
			ax.legend((lines['rewards'], _temp_line), ('Rewards', 'Episodes'), loc='upper right')
			ax.set_ylabel('RMSE')
			ax.set_xlabel('Steps')

			self.vis['fig'] = fig
			self.vis['ax'] = ax
			self.vis['lines'] = lines

		else:
			fig, ax, rew_line = self.vis['fig'], self.vis['ax'], self.vis['lines']['rewards']

			if not plt.fignum_exists(self.vis['fig'].number):
				raise Exception("Render figure was closed")

			for i, term in enumerate(self.terminal_history):
				ax.axvline(term, color='k', ls='dashed')
			self.terminal_history = []

			rew_line.set_ydata(self.reward_history)
			rew_line.set_xdata(range(len(self.reward_history)))
			ax.set_ylim((min(self.reward_history), 0))
			ax.set_xlim((0, len(self.reward_history)))
			plt.draw()

		if last:
			plt.ioff()
			plt.show(block=True)

		if nb_axes > 1:
			return self.vis['other_axes']
