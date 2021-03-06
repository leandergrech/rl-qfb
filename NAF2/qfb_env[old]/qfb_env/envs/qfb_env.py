import os
from collections import defaultdict, deque

import gym
import numpy as np
from gym.spaces import Box


class QFBEnv(gym.Env):
	# Constants
	F_s = 11245.55
	EPISODE_LENGTH_LIMIT = 70
	REWARD_SCALE = 1
	obs_lim = 2
	REWARD_DEQUE_SIZE = 5

	# Data paths
	# RM_LOC = os.path.abspath(os.path.join('../..', 'metadata', 'LHC_TRM_B1.response'))
	# CALIBRATION_LOC = os.path.abspath(os.path.join('../..', 'metadata', 'LHC_circuit.calibration'))
	# Controller boundaries in Hertz
	Q_INIT_LIMIT_HZ = 30
	Q_GOAL_HZ = 1
	Q_LIMIT_HZ = 25
	Q_STEP_MAX_HZ = 0.01 * F_s
	T_s = 1 / 12.5

	# Normalise boundaries
	Q_init_std = Q_INIT_LIMIT_HZ / Q_LIMIT_HZ
	Q_goal = Q_GOAL_HZ / Q_LIMIT_HZ
	Q_step_max = Q_STEP_MAX_HZ / Q_LIMIT_HZ

	def __init__(self, noise_std=0.0, **kwargs):
		self.RM_LOC = kwargs.get('rm_loc', 'D:/Code/rl-qfb/metadata/LHC_TRM_B1.response')
		self.CALIBRATION_LOC = kwargs.get('calibration_loc', 'D:/Code/rl-qfb/metadata/LHC_circuit.calibration')

		# self.RM_LOC = kwargs.get('rm_loc', '/mnt/d/Code/rl-qfb/metadata/LHC_TRM_B1.response')#'D:\Code\\rl-qfb\metadata\LHC_TRM_B1.response')
		# self.CALIBRATION_LOC = kwargs.get('calibration_loc', '/mnt/d/Code/rl-qfb/metadata/LHC_circuit.calibration')#'D:\Code\\rl-qfb\metadata\LHC_circuit.calibration')

		self.noise_std = noise_std

		'''QFB relate parameters'''
		self.__full_rm = None
		self.__knobNames = None
		self.__knobNormal = None
		self.circuits = None
		self.pi = None
		self.rm = None
		self.calibrations = None

		'''Setting up environment'''
		self.load_model()
		self.load_circuit_calibrations()

		'''Setting up state and action spaces'''
		self.obs_dimension = self.pi.shape[1]
		self.act_dimension = self.pi.shape[0]

		self.observation_space = Box(low=-1.0,
		                             high=1.0,
		                             shape=(self.obs_dimension,),
		                             dtype=np.float32)
		self.action_space = Box(low=-np.ones(self.act_dimension),
		                        high=np.ones(self.act_dimension),
		                        dtype=np.float32)

		'''RL related parameters'''
		self.current_state = None
		self._reward = None
		self.reward_thresh = self.objective([QFBEnv.Q_goal] * self.obs_dimension)
		self.reward_deque = deque(maxlen=QFBEnv.REWARD_DEQUE_SIZE)
		self.it = 0


	def __repr__(self):
		return f'QFBEnv'


	def reset(self, init_state=None):
		if init_state is None:
			init_state = np.random.uniform(-self.Q_init_std, self.Q_init_std, size=self.obs_dimension)
			# init_state = np.clip(a=init_state, a_min=-1.0, a_max=1.0)

		self.current_state = init_state
		self.reward_deque.clear()

		self.it = 0

		return self.current_state


	def step(self, action):
		'''
		:param action:
		:return:
		'''
		'''Calculate delta current from normalised action passed'''
		action_currents = np.array(
			[a * self.calibrations[circ]['Irate'] * self.T_s for a, circ in zip(action, self.circuits)])

		'''Get effective tune shift
			Convert from [-Q_LIMIT_HZ,Q_LIMIT_HZ] -> [-1,1]'''
		trim_state = self.rm.dot(action_currents)
		trim_state = np.divide(trim_state, self.Q_LIMIT_HZ)

		'''Update current state'''
		self.current_state += trim_state

		'''Get reward'''
		self.reward = self.objective(self.current_state)

		'''Get done signal'''
		done = self.is_done()

		return self.current_state, self.reward, done, {}


	@property
	def reward(self):
		return self._reward


	@reward.setter
	def reward(self, r):
		self._reward = r
		self.reward_deque.append(r)


	@property
	def reward_scale(self):
		return self.REWARD_SCALE


	@reward_scale.setter
	def reward_scale(self, sc):
		self.REWARD_SCALE = sc
		self.reward_thresh = self.objective([self.Q_goal] * self.obs_dimension)


	def objective(self, state):
		# state_reward = -np.square(np.sum(np.abs(state)))
		state_reward = -np.sum(np.square(state)) / self.obs_dimension

		return state_reward * self.REWARD_SCALE


	def is_done(self):
		self.it += 1

		done = False
		success = False

		# Reach goal
		if len(self.reward_deque) == QFBEnv.REWARD_DEQUE_SIZE and np.min(self.reward_deque) > self.reward_thresh:
			done = True
			success = True
		elif self.it >= self.EPISODE_LENGTH_LIMIT:
			done = True

		return done, success


	def get_optimal_action(self, state):
		state = np.multiply(state, self.Q_LIMIT_HZ)
		action_optimal = -self.pi.dot(state)
		for i, circ in enumerate(self.circuits):
			action_optimal[i] = action_optimal[i] / (self.calibrations[circ]['Irate'] * self.T_s)

		return action_optimal


	def get_knob_names(self):
		assert self.__knobNames is not None, "Environment has not been reset"
		return self.__knobNames


	def load_model(self, reload=False):
		if self.__full_rm is not None and not reload:
			return self.__full_rm

		'''Read matrix and circuit names'''
		pseudo_inverse_full = np.zeros(shape=(0, 6))  # Stores full PI from file
		circuits_full = np.array([])  # Stores all circuits from file

		def split_line_ret_floats(line):
			return [np.float32(item) for item in line.split()]

		with open(self.RM_LOC, 'r') as f:
			contents = f.readlines()
			reading_matrix = False
			reading_circuits = False
			for line in contents:
				if '# knobNames' in line:
					line = line.split(':')[1]
					self.__knobNames = [item.split('/')[1].lower() for item in line.split()]
					continue
				elif '# knobNormal' in line:
					line = line.split(':')[1]
					self.__knobNormal = split_line_ret_floats(line)
					continue

				if '# matrix' in line:
					reading_matrix = True
					reading_circuits = False
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
						pseudo_inverse_full = np.vstack([pseudo_inverse_full,
						                                 split_line_ret_floats(line)])
				elif reading_circuits:
					if '#' in line:
						reading_circuits = False
						continue
					else:
						circuits_full = np.append(circuits_full,
						                          line.lower().split('/')[0])

		'''Sanitize the matrix and circuits for use on tune feedback only'''
		circuits_tune = []
		pseudo_inverse_tune = np.zeros(shape=(0, 2))
		for i, (circ, pi_row) in enumerate(zip(circuits_full, pseudo_inverse_full[:, :2])):
			if np.sum(np.abs(pi_row)) > 0.0:  # Only useful rows are non-empty in tune columns
				circuits_tune.append(circ)
				pseudo_inverse_tune = np.vstack([pseudo_inverse_tune, pi_row])

		self.circuits = circuits_tune
		self.pi = pseudo_inverse_tune
		self.rm = np.linalg.pinv(pseudo_inverse_tune)


	def load_circuit_calibrations(self):
		"""
		Load circuit calibration data. Used to obtain maximum allowed current change
		:param circuit_calibration_file:
		:return: {device_name:{"circuit_name":..., "bnom":..., etc.}}
		"""
		calib_file = os.path.abspath(self.CALIBRATION_LOC)
		calibrations = defaultdict(dict)

		with open(calib_file) as f:
			keywords = None
			# Read and parse file
			parser = {}
			keywords = None
			for line in f:
				if '#' in line:
					# KEYWORDS --> Occurrence of # means the keywords are on this line
					keywords = [k for k in line.split() if ('|' not in k) and
					            ('#' not in k) and
					            ('circuit_name' not in k)]
					for k in keywords:
						if 'name' in k:
							parser[k] = lambda x: x.strip('"').lower()
						elif any(integer_col_hint in k for integer_col_hint in ["beam", "IR", "FGC", "idnbr"]):
							parser[k] = lambda x: int(x)
						else:
							parser[k] = lambda x: np.float32(x)
				else:
					if keywords is None:
						raise Exception("Keywords were not found at the top of file.")

					split_line = np.array([item for item in line.split() if '|' not in item])
					circ = split_line[1].lower()

					if circ not in self.circuits:
						continue
					split_line = np.concatenate([[split_line[0]], split_line[2:]])

					for i, (k, item) in enumerate(zip(keywords, split_line)):
						calibrations[circ][k] = parser[k](item)

		self.calibrations = calibrations


class QFBGoalEnv(QFBEnv):
	def __init__(self):
		super(QFBGoalEnv, self).__init__()
		self.observation_space = gym.spaces.Dict({
			'observation': Box(low=-1.0, high=1.0, shape=(self.obs_dimension,), dtype=np.float32),
			'desired_goal': Box(low=-1.0, high=1.0, shape=(self.obs_dimension,), dtype=np.float32),
			'achieved_goal': Box(low=-1.0, high=1.0, shape=(self.obs_dimension,), dtype=np.float32)
		})
		self.desired_goal = np.zeros(self.obs_dimension)

	def reset(self):
		init_obs = super(QFBGoalEnv, self).reset()

		return {'observation': init_obs,
		        'achieved_goal': init_obs,
		        'desired_goal': self.desired_goal}

	def compute_reward(self, achieved_goal, desired_goal, info):
		difference_objective = self.objective(achieved_goal - desired_goal)

		if np.abs(difference_objective) <= self.Q_GOAL_HZ / (self.F_s * self.obs_norm):
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

	fig, (ax1, ax2, ax3) = plt.subplots(3, num=1, gridspec_kw={'height_ratios': [2, 2, 1]})
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
		ax3.set_xlim((0, i + 1))

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


import matplotlib.pyplot as plt

if __name__ == '__main__':
	env = QFBEnv()
	o = env.reset()
	a = env.action_space.sample()
	o1 = env.step(a)[0]

	fig, ax = plt.subplots()
	ax.plot(o)
	ax.plot(o1)
	plt.show()
