import os
import pickle
import sys
from datetime import datetime
from datetime import datetime as dt

import gym
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import stable_baselines
from stable_baselines.common.callbacks import CheckpointCallback, BaseCallback
# from inverted_pendulum import PendulumEnv

'''This script includes the AE-DYNA algorithm. It runs on tensorflow 1.15 since it needs also the stable base lines.
https://github.com/hill-a/stable-baselines '''

# set random seed
random_seed = 123
np.random.seed(random_seed)

############################################################
# Hyperparameters
############################################################

steps_per_epoch = 1000
init_random_steps = 2000
num_epochs = 20

max_training_iterations = 30
delay_before_convergence_check = 5

simulated_steps = 5000

model_batch_size = 100
num_ensemble_models =5

early_stopping = True
model_iter = 15

network_size = 100

# How often to check the progress of the network training
# e.g. lambda it, episode: (it + 1) % max(3, (ep+1)*2) == 0
# dynamic_wait_time = lambda it, ep: (it + 1) % (ep + 1) * 2 == 0  #
dynamic_wait_time = lambda it, ep: (it + 1) % (ep + 1) == 0  # LG [19/06/2021]

# Learning rate as function of ep:
# lr_start = 5e-4
# lr_end = 5e-4
# lr = lambda ep: max(lr_start + ep / 30 * (lr_end - lr_start), lr_end)
lr = lambda ep: 1e-3

# Set max episode length manually here for the pendulum
max_steps = 200

class TrajectoryBuffer():
    '''Class for data storage during the tests'''

    def __init__(self, name, directory):
        self.save_frequency = 100000
        self.directory = directory
        self.name = name
        self.rews = []
        self.obss = []
        self.acts = []
        self.dones = []
        self.info = ""
        self.idx = -1

    def new_trajectory(self, obs):
        self.idx += 1
        self.rews.append([])
        self.acts.append([])
        self.obss.append([])
        self.dones.append([])
        self.store_step(obs=obs)

    def store_step(self, obs=None, act=None, rew=None, done=None):
        self.rews[self.idx].append(rew)
        self.obss[self.idx].append(obs)
        self.acts[self.idx].append(act)
        self.dones[self.idx].append(done)

        if self.__len__() % self.save_frequency == 0:
            self.save_buffer()

    def __len__(self):
        assert (len(self.rews) == len(self.obss) == len(self.acts) == len(self.dones))
        return len(self.obss)

    def save_buffer(self, **kwargs):
        if 'info' in kwargs:
            self.info = kwargs.get('info')
        now = datetime.now()
        # clock_time = "{}_{}_{}_{}_".format(now.day, now.hour, now.minute, now.second)
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}_'
        data = dict(obss=self.obss,
                    acts=self.acts,
                    rews=self.rews,
                    dones=self.dones,
                    info=self.info)
        # print('saving...', data)
        out_put_writer = open(self.directory + clock_time + self.name, 'wb')
        pickle.dump(data, out_put_writer, -1)
        # pickle.dump(self.actions, out_put_writer, -1)
        out_put_writer.close()

    def get_data(self):
        return dict(obss=self.obss,
                    acts=self.acts,
                    rews=self.rews,
                    dones=self.dones,
                    info=self.info)


class MonitoringEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information for scaling to correct space and for post analysis.
    '''

    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)
        self.data_dict = dict()
        self.environment_usage = 'default'
        self.directory = project_directory
        self.data_dict[self.environment_usage] = TrajectoryBuffer(name=self.environment_usage,
                                                                  directory=self.directory)
        self.current_buffer = self.data_dict.get(self.environment_usage)

        self.test_env_flag = False

        self.obs_dim = self.env.observation_space.shape
        self.obs_high = self.env.observation_space.high
        self.obs_low = self.env.observation_space.high
        self.act_dim = self.env.action_space.shape
        self.act_high = self.env.action_space.high
        self.act_low = self.env.action_space.low

        # state space definition
        self.observation_space = gym.spaces.Box(low=-1.0,
                                                high=1.0,
                                                shape=self.obs_dim,
                                                dtype=np.float64)

        # action space definition
        self.action_space = gym.spaces.Box(low=-1.0,
                                           high=1.0,
                                           shape=self.act_dim,
                                           dtype=np.float64)

        # if 'test_env' in kwargs:
        #     self.test_env_flag = True
        self.verification = False
        if 'verification' in kwargs:
            self.verification = kwargs.get('verification')
        try:
            self.max_steps = env.max_steps
        except:
            self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        init_obs = self.env.reset(**kwargs)
        self.current_buffer.new_trajectory(init_obs)
        init_obs = self.scale_state_env(init_obs)
        return init_obs

    def step(self, action):
        action = self.descale_action_env(action)
        ob, reward, done, info = self.env.step(action)
        self.current_buffer.store_step(obs=ob, act=action, rew=reward, done=done)
        ob = self.scale_state_env(ob)
        reward = self.rew_scale(reward)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return ob, reward, done, info

    def set_usage(self, usage):
        self.environment_usage = usage
        if usage in self.data_dict:
            self.current_buffer = self.data_dict.get(usage)
        else:
            self.data_dict[self.environment_usage] = TrajectoryBuffer(name=self.environment_usage,
                                                                      directory=self.directory)
            self.current_buffer = self.data_dict.get(usage)

    def close_usage(self, usage):
        # Todo: Implement to save complete data
        self.current_buffer = self.data_dict.get(usage)
        self.current_buffer.save_buffer()

    def scale_state_env(self, ob):
        return ob / 5
        # scale = (self.env.observation_space.high - self.env.observation_space.low)
        # return (2 * ob - (self.env.observation_space.high + self.env.observation_space.low)) / scale
        # return ob

    def descale_action_env(self, act):
        # scale = (self.env.action_space.high - self.env.action_space.low)
        # return np.squeeze(scale * act + self.env.action_space.high + self.env.action_space.low) / 2
        return act

    def rew_scale(self, rew):
        # we only scale for the network training:
        # if not self.test_env_flag:
        #     rew = rew * 2 + 1

        # if not self.verification:
        #     '''Rescale reward from [-1,0] to [-1,1] for the training of the network in case of tests'''
        #     rew = rew * 2 + 1
        #     pass

        #     if rew < -1:
        #         print('Hallo was geht: ', rew)
        #     else:
        #         print('Okay...', rew)
        return rew / 3

    def save_current_buffer(self, info=''):
        self.current_buffer = self.data_dict.get(self.environment_usage)
        self.current_buffer.save_buffer(info=info)
        print('Saved current buffer', self.environment_usage)

    def set_directory(self, directory):
        self.directory = directory


def flatten_list(tensor_list):
    '''
    Flatten a list of tensors
    '''
    return tf.concat([flatten(t) for t in tensor_list], axis=0)


def flatten(tensor):
    '''
    Flatten a tensor
    '''
    return tf.reshape(tensor, shape=(-1,))


def test_agent(env_test, agent_op, num_games=10):
    '''
    Test an agent 'agent_op', 'num_games' times
    Return mean and std
    '''
    games_r = []
    games_length = []
    games_success = []
    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()
        game_length = 0
        while not d:
            try:
                a_s, _ = agent_op([o])
            except:
                a_s, _ = agent_op(o)
            a_s = np.squeeze(a_s)
            o, r, d, _ = env_test.step(a_s)
            game_r += r
            game_length += 1
            # print(o, a_s, r)
        success = r > -0.0016
        # print(r)
        games_r.append(game_r/game_length)
        games_length.append(game_length)
        games_success.append(success)
    return np.mean(games_r), np.std(games_r), np.mean(games_length), np.mean(games_success)


class FullBuffer():
    def __init__(self):
        self.rew = []
        self.obs = []
        self.act = []
        self.nxt_obs = []
        self.done = []

        self.train_idx = []
        self.valid_idx = []
        self.idx = 0

    def store(self, obs, act, rew, nxt_obs, done):
        self.rew.append(rew)
        self.obs.append(obs)
        self.act.append(act)
        self.nxt_obs.append(nxt_obs)
        self.done.append(done)

        self.idx += 1

    def generate_random_dataset(self, ratio=False):
        """ratio: how much for valid taken"""
        rnd = np.arange(len(self.obs))
        np.random.shuffle(rnd)
        self.valid_idx = rnd[:]
        self.train_idx = rnd[:]  # change back
        if ratio:
            self.valid_idx = rnd[: int(len(self.obs) * ratio)]
            self.train_idx = rnd[int(len(self.obs) * ratio):]

        print('Train set:', len(self.train_idx), 'Valid set:', len(self.valid_idx))

    def get_training_batch(self):
        return np.array(self.obs)[self.train_idx], np.array(np.expand_dims(self.act, -1))[self.train_idx], \
               np.array(self.rew)[self.train_idx], np.array(self.nxt_obs)[self.train_idx], np.array(self.done)[
                   self.train_idx]

    def get_valid_batch(self):
        return np.array(self.obs)[self.valid_idx], np.array(np.expand_dims(self.act, -1))[self.valid_idx], \
               np.array(self.rew)[self.valid_idx], np.array(self.nxt_obs)[self.valid_idx], np.array(self.done)[
                   self.valid_idx]

    def get_maximum(self):
        idx = np.argmax(self.rew)
        print('rew', np.array(self.rew)[idx])
        return np.array(self.obs)[idx], np.array(np.expand_dims(self.act, -1))[idx], \
               np.array(self.rew)[idx], np.array(self.nxt_obs)[idx], np.array(self.done)[
                   idx]

    def __len__(self):
        assert (len(self.rew) == len(self.obs) == len(self.act) == len(self.nxt_obs) == len(self.done))
        return len(self.obs)


class NN:
    '''
    Takes care of aleatoric uncertainties. Method developed at Oxford. [LG 29/04/2021]
    '''
    def __init__(self, x, y, y_dim, hidden_size, n, learning_rate, init_params):
        self.init_params = init_params

        # set up NN
        with tf.variable_scope('model_' + str(n) + '_nn'):
            self.inputs = x
            self.y_target = y

            self.inputs = tf.scalar_mul(0.5, self.inputs)
            self.layer_1_w = tf.layers.Dense(hidden_size,
                                             # activation=tf.nn.tanh,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                             stddev=self.init_params.get(
                                                                                                 'init_stddev_1_w'),
                                                                                             dtype=tf.float64),
                                             bias_initializer=tf.random_normal_initializer(mean=0.,
                                                                                           stddev=self.init_params.get(
                                                                                               'init_stddev_1_b'),
                                                                                           dtype=tf.float64))

            self.layer_1 = self.layer_1_w.apply(self.inputs)
            self.layer_1 = tf.scalar_mul(0.5, self.layer_1)
            self.layer_2_w = tf.layers.Dense(hidden_size,
                                             # activation=tf.nn.tanh,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                             stddev=self.init_params.get(
                                                                                                 'init_stddev_1_w'),
                                                                                             dtype=tf.float64),
                                             bias_initializer=tf.random_normal_initializer(mean=0.,
                                                                                           stddev=self.init_params.get(
                                                                                               'init_stddev_1_b'),
                                                                                           dtype=tf.float64))

            self.layer_2 = self.layer_2_w.apply(self.layer_1)
            #
            self.output_w = tf.layers.Dense(y_dim,
                                            activation=None,
                                            # use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                            stddev=self.init_params.get(
                                                                                                'init_stddev_2_w'),
                                                                                            dtype=tf.float64))

            self.output = self.output_w.apply(self.layer_2)

            # set up loss and optimiser - we'll modify this later with anchoring regularisation
            self.opt_method = tf.train.AdamOptimizer(learning_rate)
            self.mse_ = tf.reduce_mean(((self.y_target - self.output)) ** 2)
            self.loss_ = 1 / tf.shape(self.inputs, out_type=tf.int64)[0] * \
                         tf.reduce_sum(tf.square(self.y_target - self.output))
            self.optimizer = self.opt_method.minimize(self.loss_)
            self.optimizer_mse = self.opt_method.minimize(self.mse_)

    def get_weights(self, sess):
        """method to return current params"""

        ops = [self.layer_1_w.kernel, self.layer_1_w.bias,
               self.layer_2_w.kernel, self.layer_2_w.bias,
               self.output_w.kernel]
        w1, b1, w2, b2, w = sess.run(ops)

        return w1, b1, w2, b2, w

    def anchor(self, lambda_anchor, sess):
        """regularise around initialised parameters after session has started"""

        w1, b1, w2, b2, w = self.get_weights(sess=sess)

        # get initial params to hold for future trainings
        self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.w_out_init = w1, b1, w2, b2, w

        loss_anchor = lambda_anchor[0] * tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
        loss_anchor += lambda_anchor[1] * tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))

        loss_anchor = lambda_anchor[0] * tf.reduce_sum(tf.square(self.w2_init - self.layer_2_w.kernel))
        loss_anchor += lambda_anchor[1] * tf.reduce_sum(tf.square(self.b2_init - self.layer_2_w.bias))

        loss_anchor += lambda_anchor[2] * tf.reduce_sum(tf.square(self.w_out_init - self.output_w.kernel))

        # combine with original loss
        # TODO: commented for now
        # self.loss_ = self.loss_ + tf.scalar_mul(1 / tf.shape(self.inputs)[0], loss_anchor)
        self.optimizer = self.opt_method.minimize(self.loss_)
        return self.optimizer, self.loss_


class NetworkEnv(gym.Wrapper):
    '''
    Wrapper to handle the network interaction
    Here you can change the treatment of the uncertainty
    '''

    def __init__(self, env, model_func=None, done_func=None, number_models=1, **kwargs):
        gym.Wrapper.__init__(self, env)
        self.number_models = number_models
        self.current_model = np.random.randint(0, max(self.number_models, 1))
        self.model_func = model_func
        self.done_func = done_func

        self.len_episode = 0
        self.max_steps = env.max_steps
        self.verification = False
        if 'verification' in kwargs:
            self.verification = kwargs.get('verification')
        # self.visualize()

    def reset(self, **kwargs):
        self.current_model = np.random.randint(0, max(self.number_models, 1))
        self.len_episode = 0
        self.done = False
        # Here is a main difference to other dyna style methods:
        # obs = np.random.uniform(-1, 1, self.env.observation_space.shape)

        obs = self.env.reset()
        self.obs = np.clip(obs, -1.0, 1.0)
        return self.obs

    def step(self, action):
        if self.verification:
            obs, rew = self.model_func(self.obs, [np.squeeze(action)])
        else:
            # Can be activated to randomize each step
            current_model = np.random.randint(0, max(self.number_models, 1))  # self.current_model
            # current_model = self.current_model
            obs, rew = self.model_func(self.obs, [np.squeeze(action)], current_model)
        # obs, rew, _, _ = self.env.step(action)
        self.obs = np.clip(obs.copy(), -1, 1)
        # rew = np.clip(rew, -1, 1)
        if not self.verification:
            rew = (rew - 1) / 2
        self.len_episode += 1

        if self.len_episode >= self.max_steps:
            self.done = True

        return self.obs, rew, self.done, dict()

    def visualize(self, data=None, label=None, **kwargs):
        action = [np.zeros(self.env.action_space.shape)]
        state = np.zeros(self.env.observation_space.shape)
        maximum = 0
        if data is not None:
            # action = [data[1]]
            state = data[0]
            maximum = (data[2] - 1) / 2
        delta = 0.05
        x = np.arange(-1, 1, delta)
        y = np.arange(-1, 1, delta)
        X, Y = np.meshgrid(x, y)

        if 'data_points' in kwargs:
            data_points = kwargs.get('data_points')
        if self.number_models == num_ensemble_models:
            Nr = 1
            Nc = 1
            fig, axs = plt.subplots(Nr, Nc)
            fig.subplots_adjust(hspace=0.3)
            images = []
            for nr in range(self.number_models):
                rewards = np.zeros(X.shape)

                # print(self.number_models)
                for i1 in range(len(x)):
                    for j1 in range(len(y)):
                        state[0] = np.cos(x[i1])
                        state[1] = np.sin(x[j1])
                        state[1] = y[j1]
                        rewards[i1, j1] = (self.model_func(state, [np.squeeze(action)],
                                                           nr))[1] / num_ensemble_models
                axs.contour(X, Y, (rewards - 1) / 2, alpha=1)
                self.save_buffer(nr, data, X, Y, rewards)
            # list_combinations = list(it.combinations([0, 1, 2, 3], 2))
            #
            # for i in range(Nr):
            #     for j in range(Nc):
            #
            #         for nr in range(self.number_models):
            #             rewards = np.zeros(X.shape)
            #
            #             # print(self.number_models)
            #             for i1 in range(len(x)):
            #                 for j1 in range(len(y)):
            #                     current_pair = list_combinations[i * Nc + j]
            #                     state[current_pair[0]] = x[i1]
            #                     state[current_pair[1]] = y[j1]
            #                     rewards[i1, j1] = (self.model_func(state, [np.squeeze(action)],
            #                                                        nr))[1] / num_ensemble_models
            #             axs[i, j].contour(X, Y, (rewards - 1) / 2, alpha=1)
            #             # plt.plot(np.array(states, dtype=object)[:, 1],)
            #         # images.append(axs[i, j].contour(X, Y, (rewards - 1) / 2, 25, alpha=1))
            #         # axs[i, j].label_outer()

            plt.title(maximum)
            # plt.title(label)
            # plt.colorbar()
            fig.show()
        else:
            pass
            # action = [np.random.uniform(-1, 1, 4)]
            # state_vec = np.linspace(-1, 1, 100)
            # states = []
            # # print(self.number_models)
            #
            # for i in state_vec:
            #     states.append(self.model_func(np.array([i, 0, 0, 0]), action,
            #                                   self.number_models))
            #
            # plt.plot(np.array(states, dtype=object)[:, 1])

            # states = np.zeros(X.shape)
            # # print(self.number_models)
            # for i in range(len(x)):
            #     for j in range(len(y)):
            #         states[i, j] = (self.model_func(np.array([x[i], y[j], 0, 0]), action,
            #                                         self.number_models)[1])
            # plt.contourf(states)

    def save_buffer(self, model_nr, data, X, Y, rews, **kwargs):
        if 'info' in kwargs:
            self.info = kwargs.get('info')
        now = datetime.now()
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}_'
        data = dict(data=data,
                    model=model_nr,
                    rews=rews,
                    X=X,
                    Y=Y)
        out_put_writer = open(project_directory + clock_time + 'plot_model_' + str(model_nr), 'wb')
        pickle.dump(data, out_put_writer, -1)
        out_put_writer.close()


class StructEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information like number of steps and total reward of the last espisode.
    '''

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n_obs = self.env.reset()
        self.total_rew = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.n_obs = self.env.reset(**kwargs)
        self.total_rew = 0
        self.len_episode = 0
        return self.n_obs.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # print('reward in struct', reward)
        self.total_rew += reward
        self.len_episode += 1
        return ob, reward, done, info

    def get_episode_reward(self):
        return self.total_rew

    def get_episode_length(self):
        return self.len_episode


def restore_model(old_model_variables, m_variables):
    # variable used as index for restoring the actor's parameters
    it_v2 = tf.Variable(0, trainable=False)

    restore_m_params = []
    for m_v in m_variables:
        upd_m_rsh = tf.reshape(old_model_variables[it_v2: it_v2 + tf.reduce_prod(m_v.shape)], shape=m_v.shape)
        restore_m_params.append(m_v.assign(upd_m_rsh))
        it_v2 += tf.reduce_prod(m_v.shape)

    return tf.group(*restore_m_params)


class EvaluationCallback(BaseCallback):
	MAX_EPS = 20
	def __init__(self, env):
		self.env = env
		self.current_best_model_ep_len = self.env.EPISODE_LENGTH_LIMIT
		self.current_best_model_save_dir = ''

		self.gamma = 0.99
		self.discounts = [np.power(self.gamma, i) for i in range(self.env.EPISODE_LENGTH_LIMIT)]
		super(EvaluationCallback, self).__init__()

	def _on_step(self):
		# if self.locals['done']:
		# 	if self.env.it >= self.env.max_steps:
		# 		self.locals['done'] = False
		# 		self.env.reset()

		if self.num_timesteps % 1000 == 0:
			q_err = []
			returns = []
			ep_lens = []
			success = []

			for ep in range(self.MAX_EPS):
				o = self.env.reset()
				step = 0

				ep_state_list = []
				ep_action_list = []
				ep_rewards_list = []

				while True:
					step += 1
					a = self.model.predict(o, deterministic=True)[0]

					ep_state_list.append(o)
					ep_action_list.append(a)

					o, r, d, _ = self.env.step(a)

					ep_rewards_list.append(r)
					if d:
						break

					### END OF EPISODE LOOP ###

				# Measure q estimation error
				# estimated_q = self.model.sess.run(self.model.target_policy_tf.qf1,
				# 								  {self.model.actions_ph: ep_action_list,
				# 								   self.model.processed_next_obs_ph: ep_state_list}).reshape(-1)
				# real_q = []
				# for i in range(step):
				# 	real_q.append(sum([rew * disc for rew, disc in zip(ep_rewards_list[i:], self.discounts)]))
				# q_err.append(np.mean(np.subtract(estimated_q, real_q)))

				ep_lens.append(step)
				returns.append(sum(ep_rewards_list))
				if ep_lens[-1] == self.env.EPISODE_LENGTH_LIMIT:
					success.append(0.0)
				else:
					success.append(1.0)

				### END OF EPISODE ###

			# q_err = np.mean(q_err)
			returns = np.mean(returns)
			ep_lens = np.mean(ep_lens)
			success = np.mean(success) * 100

			for tag, val in zip(('episode_return', 'episode_length', 'success'),#, 'q_err'),
								(returns, ep_lens, success)):#, q_err)):
				summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
				self.locals['writer'].add_summary(summary, self.num_timesteps)

		return True


def aedyna(real_env, eval_env, num_epochs=50, steps_per_env=100, algorithm='SAC',
           simulated_steps=1000, num_ensemble_models=2, model_iter=15, model_batch_size=512,
           init_random_steps=steps_per_epoch, agent_params={}):
    '''
    Anchor ensemble dyna reinforcement learning

    The states and actions are provided by the gym environment with the correct boxes.

    Parameters:
    -----------
    real_env: Environment
    num_epochs: number of training epochs
    steps_per_env: number of steps per environment
            # NB: the total number of steps per epoch will be: steps_per_env*number_envs
    algorithm: type of algorithm. Either 'PPO' or 'SAC'
    minibatch_size: Batch size used to train the critic
    mb_lr: learning rate of the environment model
    model_batch_size: batch size of the environment model
    simulated_steps: number of simulated steps for each policy update
    model_iter: number of iterations without improvement before stopping training the model
    '''

    model_name = f'AE-DYNA-{algorithm}_{dt.strftime(dt.now(), "%m%d%y_%H%M")}'
    save_dir = os.path.join(f'models_{algorithm}', model_name)

    callback_chkpt = CheckpointCallback(save_freq=100, save_path=save_dir, name_prefix=model_name)
    eval_callback = EvaluationCallback(eval_env)
    agent_callbacks = [callback_chkpt, eval_callback]

    # Select the RL-algorithm
    if algorithm == 'PPO':
        from stable_baselines.common.policies import MlpPolicy
        from stable_baselines import PPO2 as Agent
    elif algorithm == 'SAC':
        from stable_baselines.sac.policies import MlpPolicy
        from stable_baselines import SAC as Agent
    else:
        assert False, f"Either 'SAC' or 'PPO': {algorithm} not recognised"

    tf.reset_default_graph()

    def make_env(**kwargs):
        '''Create the environement'''
        return MonitoringEnv(env=real_env, **kwargs)

    try:
        env_name = real_env.__name__
    except:
        env_name = 'default'

    # Create a few environments to collect the trajectories
    env = StructEnv(make_env())
    env_test = StructEnv(make_env(verification=True))

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Placeholders for model
    act_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float64, name='act')
    obs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float64, name='obs')
    # NEW
    nobs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float64, name='nobs')
    rew_ph = tf.placeholder(shape=(None, 1), dtype=tf.float64, name='rew')

    # Placeholder for learning rate
    mb_lr_ = tf.placeholder("float", None)

    old_model_variables = tf.placeholder(shape=(None,), dtype=tf.float64, name='old_model_variables')

    def variables_in_scope(scope):
        # get all trainable variables in 'scope'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    #########################################################
    ######################### MODEL #########################
    #########################################################

    m_opts = []
    m_losses = []

    nobs_pred_m = []
    act_obs = tf.concat([obs_ph, act_ph], 1)
    target = tf.concat([nobs_ph, rew_ph], 1)

    # computational graph of N models and the correct losses for the anchor method
    m_classes = []

    for i in range(num_ensemble_models):
        m_class = NN(x=act_obs, y=target, y_dim=obs_dim[0] + 1,
                     learning_rate=mb_lr_, n=i,
                     hidden_size=network_size, init_params=init_params)

        nobs_pred = m_class.output

        nobs_pred_m.append(nobs_pred)

        m_classes.append(m_class)
        m_losses.append(m_class.mse_)
        m_opts.append(m_class.optimizer_mse)

    ##################### RESTORE MODEL ######################
    initialize_models = []
    models_variables = []
    for i in range(num_ensemble_models):
        m_variables = variables_in_scope('model_' + str(i) + '_nn')
        initialize_models.append(restore_model(old_model_variables, m_variables))
        #  List of weights
        models_variables.append(flatten_list(m_variables))

    #########################################################
    ##################### END MODEL #########################
    #########################################################
    file_writer_model = tf.summary.FileWriter(os.path.join('logs_model', model_name), tf.get_default_graph())
    agent_tb_loc = os.path.join('logs_agent', model_name)

    #################################################################################################

    # Tensorflow session start!!!!!!!!
    # Create a session
    try:
        sess = tf.Session(config=config)
    except:
        config = None
        sess = tf.Session(config=config)
    # Initialize the variables
    sess.run(tf.global_variables_initializer())

    def model_op(o, a, md_idx):
        """Calculate the predictions of the dynamics model"""
        # mo = sess.run(nobs_pred_m[md_idx], feed_dict={obs_ph: [o], act_ph: [a]})
        mo = sess.run(nobs_pred_m[md_idx], feed_dict={obs_ph: [o], act_ph: a})
        return np.squeeze(mo[:, :-1]), float(np.squeeze(mo[:, -1]))

    def run_model_loss(model_idx, r_obs, r_act, r_nxt_obs, r_rew):
        # TODO: Uncommented line below
        r_act = np.squeeze(r_act, axis=2)
        r_rew = np.reshape(r_rew, (-1, 1))
        return_val = sess.run(m_loss_anchor[model_idx],
                              feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs, rew_ph: r_rew})
        return return_val

    def run_model_opt_loss(model_idx, r_obs, r_act, r_nxt_obs, r_rew, mb_lr):
        r_act = np.squeeze(r_act, axis=2)
        r_rew = np.reshape(r_rew, (-1, 1))
        return sess.run([m_opts_anchor[model_idx], m_loss_anchor[model_idx]],
                        feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs, rew_ph: r_rew, mb_lr_: mb_lr})

    def model_assign(i, model_variables_to_assign):
        '''
        Update the i-th model's parameters
        '''
        return sess.run(initialize_models[i], feed_dict={old_model_variables: model_variables_to_assign})

    def train_model(tr_obs, tr_act, tr_nxt_obs, tr_rew, v_obs, v_act, v_nxt_obs, v_rew, step_count, model_idx, mb_lr):

        # Get validation loss on the old model only used for monitoring
        mb_valid_loss1 = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

        # Restore the initial random weights to have a new, clean neural network
        # initial_variables_models - list stored before already in the code below -
        # important for the anchor method
        model_assign(model_idx, initial_variables_models[model_idx])

        # Get validation loss on the now initialized model
        mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

        acc_m_losses = []

        md_params = sess.run(models_variables[model_idx])
        best_mb = {'iter': 0, 'loss': mb_valid_loss, 'params': md_params}
        it = 0

        # Create mini-batch for training
        lb = len(tr_obs)

        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        if not early_stopping:
            # model_batch_size = lb
            # Take a fixed accuracy
            not_converged = True
            while not_converged:

                # update the model on each mini-batch
                last_m_losses = []
                for idx in range(0, lb, lb):
                    minib = shuffled_batch

                    _, ml = run_model_opt_loss(model_idx, tr_obs[minib], tr_act[minib], tr_nxt_obs[minib],
                                               tr_rew[minib], mb_lr=mb_lr)
                    acc_m_losses.append(ml)
                    last_m_losses.append(ml)
                    mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

                    if mb_valid_loss < max(mb_lr, 1e-4) or it > 1e5:
                        not_converged = False
                    it += 1

            best_mb['loss'] = mb_valid_loss
            best_mb['iter'] = it
            # store the parameters to the array
            best_mb['params'] = sess.run(models_variables[model_idx])

        else:
            # Run until the number of model_iter has passed from the best val loss at it on...
            # ml = 1
            # while not (best_mb['iter'] < it - model_iter and ml < 5e-3):
            while best_mb['iter'] > it - model_iter:
                # update the model on each mini-batch
                last_m_losses = []
                for idx in range(0, lb, model_batch_size):
                    minib = shuffled_batch[idx:min(idx + model_batch_size, lb)]
                    _, ml = run_model_opt_loss(model_idx, tr_obs[minib], tr_act[minib], tr_nxt_obs[minib],
                                               tr_rew[minib], mb_lr=mb_lr)
                    acc_m_losses.append(ml)
                    last_m_losses.append(ml)

                # Check if the loss on the validation set has improved
                mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

                if mb_valid_loss < best_mb['loss']:
                    best_mb['loss'] = mb_valid_loss
                    best_mb['iter'] = it
                    # store the parameters to the array
                    best_mb['params'] = sess.run(models_variables[model_idx])

                it += 1

        # Restore the model with the lower validation loss
        model_assign(model_idx, best_mb['params'])

        print(f"Model:{model_idx}, iter:{it} -- Old Val loss:{mb_valid_loss1:.6f}  New Val loss:{best_mb['loss']:.6f} -- "
              f"New Train loss:{np.mean(last_m_losses):.6f} -- Loss_data {ml:.6f}")

        summary = tf.Summary()
        summary.value.add(tag='supplementary/m_loss', simple_value=np.mean(acc_m_losses))
        summary.value.add(tag='supplementary/iterations', simple_value=it)
        file_writer_model.add_summary(summary, step_count)
        file_writer_model.flush()

    def plot_results(env_wrapper, label=None, **kwargs):
        """ Plot the validation episodes"""
        rewards = env_wrapper.env.current_buffer.get_data()['rews']

        iterations = []
        finals = []
        inits = []
        # means = []
        # stds = []

        for i in range(len(rewards)):
            if (len(rewards[i]) > 1):
                finals.append(rewards[i][-1])
                # means.append(np.mean(rewards[i][1:]))
                # stds.append(np.std(rewards[i][1:]))
                inits.append(rewards[i][0])
                iterations.append(len(rewards[i]))
        x = range(len(iterations))
        iterations = np.array(iterations)
        finals = np.array(finals)
        inits = np.array(inits)
        # means = np.array(means)
        # stds = np.array(stds)

        plot_suffix = label

        fig, axs = plt.subplots(2, 1, sharex=True)

        ax = axs[0]
        ax.plot(x, iterations)
        ax.set_ylabel('Iterations (1)')
        ax.set_title(plot_suffix)

        if 'data_number' in kwargs:
            ax1 = plt.twinx(ax)
            color = 'lime'
            ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(x, kwargs.get('data_number'), color=color)

        ax = axs[1]
        color = 'blue'
        ax.set_ylabel('Final reward', color=color)  # we already handled the x-label with ax1
        ax.tick_params(axis='y', labelcolor=color)
        ax.plot(x, finals, color=color)

        ax.set_title('Final reward per episode')  # + plot_suffix)
        ax.set_xlabel('Episodes (1)')

        ax1 = plt.twinx(ax)
        color = 'lime'
        ax1.set_ylabel('Init reward', color=color)  # we already handled the x-label with ax1
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.fill_between(x, means - stds, means + stds,
        #                  alpha=0.5, edgecolor=color, facecolor='#FF9848')
        # ax1.plot(x, means, color=color)
        ax1.plot(x, inits, color=color)

        if 'save_name' in kwargs:
            plt.savefig(kwargs.get('save_name') + '.pdf')
        plt.show()

    def plot_observables(data, label, **kwargs):
        """plot observables during the test"""

        sim_rewards_all = np.array(data.get('sim_rewards_all'))
        step_counts_all = np.array(data.get('step_counts_all'))
        batch_rews_all = np.array(data.get('batch_rews_all'))
        tests_all = np.array(data.get('tests_all'))

        x = np.arange(len(batch_rews_all[0]))

        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.suptitle(label)

        ax = axs[0]
        color = 'b'
        ax.plot(x, batch_rews_all[0], color=color, label='Batch rewards')
        ax.fill_between(x, batch_rews_all[0] - batch_rews_all[1], batch_rews_all[0] + batch_rews_all[1],
                        alpha=0.5, color=color)
        ax.set_ylabel('Rewards')
        ax.legend(loc='upper left')

        ax2 = ax.twinx()
        color = 'lime'
        ax2.step(x, step_counts_all, color=color, label='step_counts_all')
        ax2.set_ylabel('data points', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        ax = axs[1]
        # ax.set_yscale('symlog')
        color = 'b'
        ax.plot(sim_rewards_all[0], ls=':', color=color, label='Simulation rewards')
        ax.fill_between(x, sim_rewards_all[0] - sim_rewards_all[1], sim_rewards_all[0] + sim_rewards_all[1],
                        alpha=0.5, color=color)
        try:
            color = 'red'
            ax.plot(tests_all[0], color=color, label='Test rewards')
            ax.fill_between(x, tests_all[0] - tests_all[1], tests_all[0] + tests_all[1],
                            color=color, alpha=0.5)
            # ax.axhline(y=np.max(tests_all[0]), c='orange')
        except:
            pass
        ax.set_ylabel('Rewards')
        # plt.tw
        ax.grid(True)
        ax.legend(loc='upper left')

        ax2 = ax.twinx()
        color = 'lime'
        ax2.plot(length_all, color=color, label='Test')

        ax2.set_ylabel('Episode length', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)

        ax2.legend(loc='upper right')

        fig.align_labels()
        fig.tight_layout()
        plt.show()

    def save_data(data, **kwargs):
        '''logging function to save results to pickle'''
        now = datetime.now()
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}'
        with open(project_directory + clock_time + '_training_observables', 'wb') as f:
            pickle.dump(data, f, -1)

    # variable to store the total number of steps
    step_count = 0
    model_buffer = FullBuffer()
    print('Env batch size:', steps_per_env, ' Batch size:', steps_per_env)

    # Create a simulated environment
    sim_env = NetworkEnv(make_env(), model_op, None, num_ensemble_models)

    # ------------------------------------------------------------------------------------------------------
    # -------------------------------------Set correct anchors---------------------------------------
    # Get the initial parameters of each model
    # These are used in later epochs when we aim to re-train the models anew with the new dataset
    initial_variables_models = []
    for model_var in models_variables:
        initial_variables_models.append(sess.run(model_var))

    # update the anchor model losses:
    m_opts_anchor = []
    m_loss_anchor = []
    for i in range(num_ensemble_models):
        opt, loss = m_classes[i].anchor(lambda_anchor=lambda_anchor, sess=sess)
        m_opts_anchor.append(opt)
        m_loss_anchor.append(loss)

    # ------------------------------------------------------------------------------------------------------

    total_iterations = 0

    sim_rewards_all = []
    sim_rewards_std_all = []
    length_all = []
    tests_all = []
    tests_std_all = []
    batch_rews_all = []
    batch_rews_std_all = []
    step_counts_all = []

    agent = Agent(MlpPolicy, sim_env, **agent_params, tensorboard_log=agent_tb_loc, verbose=1)
    for ep in range(num_epochs):

        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []
        print(f'============================ Epoch {ep} ============================')
        # Execute in serial the environment, storing temporarily the trajectories.
        env.reset()

        # iterate over a fixed number of steps
        steps_train = init_random_steps if ep == 0 else steps_per_env

        print(f'-> Get data from {steps_train} steps')
        for _ in range(steps_train):
            # run the policy
            if ep == 0:
                # Sample random action during the first epoch
                act = np.random.uniform(-1, 1, size=act_dim)
                act = np.squeeze(act)
            else:
                noise = 0.01 * np.random.randn(act_dim)
                act, _ = agent.predict(env.n_obs)
                act = np.clip(np.squeeze(act) + noise, -1, 1)
            # take a step in the environment
            obs2, rew, done, _ = env.step(act)
            # add the new transition to the temporary buffer
            model_buffer.store(env.n_obs.copy(), act, rew.copy(), obs2.copy(), done)

            env.n_obs = obs2.copy()
            step_count += 1

            if done:
                batch_rew.append(env.get_episode_reward() / env.get_episode_length())
                batch_len.append(env.get_episode_length())

                env.reset()

        # save the data for plotting the collected data for the model
        env.save_current_buffer()

        print(f' `->Ep:{ep} Rew:{np.mean(batch_rew):.2f} -- Step:{step_count}')

        ############################################################
        ###################### MODEL LEARNING ######################
        ############################################################

        target_threshold = max(model_buffer.rew)
        sim_env.threshold = target_threshold  # min(target_threshold, -0.05)
        # print('maximum: ', sim_env.threshold)

        mb_lr = lr(ep)
        # print('mb_lr: ', mb_lr)
        if early_stopping:
            model_buffer.generate_random_dataset(ratio=0.1)
        else:
            model_buffer.generate_random_dataset()

        print(f'-> Train {num_ensemble_models} models')
        for i in range(num_ensemble_models):
            # Initialize randomly a training and validation set

            # get both datasets
            train_obs, train_act, train_rew, train_nxt_obs, _ = model_buffer.get_training_batch()
            valid_obs, valid_act, valid_rew, valid_nxt_obs, _ = model_buffer.get_valid_batch()
            # train the dynamic model on the datasets just sampled
            train_model(train_obs, train_act, train_nxt_obs, train_rew, valid_obs, valid_act, valid_nxt_obs, valid_rew,
                        step_count, i, mb_lr=mb_lr)

        ############################################################
        ###################### POLICY LEARNING #####################
        ############################################################
        print(f'-> Train {algorithm} agent on models')
        data = model_buffer.get_maximum()
        label = f'Total {total_iterations}, ' + \
                f'data points: {len(model_buffer)}, ' + \
                f'ep: {ep}, max: {data}\n' + hyp_str_all
        # sim_env.visualize(data=data, label=label)

        best_sim_test = -1e16 * np.ones(num_ensemble_models)
        agent = Agent(MlpPolicy, sim_env, **agent_params, tensorboard_log=agent_tb_loc, verbose=1)
        for it in range(max_training_iterations):
            total_iterations += 1
            print(' `-> Policy it', it, end='..')

            ################# Agent UPDATE ################
            agent.learn(total_timesteps=simulated_steps, tb_log_name=model_name, log_interval=10,
                        reset_num_timesteps=False, callback=agent_callbacks)

            if dynamic_wait_time(it, ep):
            # if True:
                print(' `-> Iterations: ', total_iterations)
                label = f'Total {total_iterations}, ' + \
                        f'data points: {len(model_buffer)}, ' + \
                        f'ep: {ep}, it: {it}\n' + hyp_str_all
                env_test.env.set_usage('test')
                mn_test, mn_test_std, mn_length, mn_success = test_agent(env_test, agent.predict, num_games=5)
                print(f' `-> Test score: Mean rew={mn_test:.2f} Std rew={mn_test_std:.2f} Mean len={mn_length:.2f} success={mn_success * 100:.2f}')

                # Save the data for plotting the tests
                tests_all.append(mn_test)
                tests_std_all.append(mn_test_std)
                length_all.append(mn_length)

                env_test.env.set_usage('default')

                print(' `-> Simulated score:')

                sim_rewards = []
                for i in range(num_ensemble_models):
                    sim_m_env = NetworkEnv(make_env(), lambda o, a: model_op(o, a, i), None, number_models=i,
                                           verification=True)
                    mn_sim_rew, mn_sim_rew_std, mn_sim_len, mn_sim_succ = test_agent(sim_m_env, agent.predict, num_games=10)
                    sim_rewards.append(mn_sim_rew)

                    print(
                        f'   `-> Model {i}: Mean rew={mn_sim_rew:.2f} Std rew={mn_sim_rew_std:.2f} Mean len={mn_sim_len:.2f} success={mn_sim_succ * 100:.2f}')

                step_counts_all.append(step_count)

                sim_rewards = np.array(sim_rewards)
                sim_rewards_all.append(np.mean(sim_rewards))
                sim_rewards_std_all.append(np.std(sim_rewards))

                batch_rews_all.append(np.mean(batch_rew))
                batch_rews_std_all.append(np.std(batch_rew))

                data = dict(sim_rewards_all=[sim_rewards_all, sim_rewards_std_all],
                            entropy_all=length_all,
                            step_counts_all=step_counts_all,
                            batch_rews_all=[batch_rews_all, batch_rews_std_all],
                            tests_all=[tests_all, tests_std_all],
                            info=label)

                # save the data for plotting the progress -------------------
                # save_data(data=data)

                # plotting the progress -------------------
                # if it % 10 == 0:
                plot_observables(data=data, label=label)

                # stop training if the policy hasn't improved
                if (np.sum(best_sim_test >= sim_rewards) >= int(num_ensemble_models * 0.7)):
                    if it > delay_before_convergence_check and ep < num_epochs - 1:
                        print(f'-> BREAK - no improvement in {int(num_ensemble_models * 0.7)} models')
                        break
                else:
                    best_sim_test = sim_rewards

    # Final verification:
    env_test.env.set_usage('final')
    mn_test, mn_test_std, mn_length, _ = test_agent(env_test, agent.predict, num_games=50)

    label = f'Verification : total {total_iterations}, ' + \
            f'data points: {len(model_buffer.train_idx) + len(model_buffer.valid_idx)}, ' + \
            f'ep: {ep}, it: {it}\n' + \
            f'rew: {mn_test}, std: {mn_test_std}'
    plot_results(env_test, label=label)

    env_test.save_current_buffer(info=label)

    env_test.env.set_usage('default')

    # closing environments..
    env.close()
    file_writer_model.close()


def old_main():
    try:
        random_seed = int(sys.argv[2])
    except:
        random_seed = 25
    try:
        file_name = sys.argv[1] + '_' + str(random_seed)
    except:
        file_name = 'defaultexp_noise_' + str(random_seed) + '_'
    # set random seed
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    try:
        root_dir = sys.argv[3]
    except:
        root_dir = 'Data/Simulation/'

    directory = root_dir + file_name + '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        # clipped_double_q
        index = int(sys.argv[4])
        parameter_list = [
            dict(noise=0.0, data_noise=0, models=1),
            dict(noise=0.0, data_noise=0, models=3),
            dict(noise=0.0, data_noise=0, models=5),
            dict(noise=0.0, data_noise=0, models=10),
            # dict(noise=0.05, data_noise=0),
            # dict(noise=0.05, data_noise=0.05),

        ]
        parameters = parameter_list[index]
        print('Running...', parameters)
    except:
        parameters = dict(noise=0.05, data_noise=0.0, models=num_ensemble_models)
        print('Running default...', parameters)

    directory = root_dir + file_name + '/'

    # Create the logging directory:
    project_directory = directory#'Data_logging/Simulation/'

    num_ensemble_models = parameters.get('models')
    hyp_str_all = 'nr_steps_' + str(steps_per_epoch) + '-n_ep_' + str(num_epochs) + \
                  '-m_bs_' + str(model_batch_size) + \
                  '-sim_steps_' + str(simulated_steps) + \
                  '-m_iter_' + str(model_iter) + '-ensnr_' + str(num_ensemble_models) + '-init_' + str(
        init_random_steps) + '/'
    project_directory = project_directory + hyp_str_all

    # To label the plots:
    hyp_str_all = '-nr_steps_' + str(steps_per_epoch) + '-n_ep_' + str(num_epochs) + \
                  '-m_bs_' + str(model_batch_size) + \
                  '-sim_steps_' + str(simulated_steps) + \
                  '-m_iter_' + str(model_iter) + \
                  '\n-ensnr_' + str(num_ensemble_models)

    if not os.path.isdir(project_directory):
        os.makedirs(project_directory)
        print("created folder : ", project_directory)

    ############################################################
    # Loading the environment
    ############################################################
    class TestWrapperEnv(gym.Wrapper):
        """
        Gym Wrapper to add noise and visualise.
        """

        def __init__(self, env, render=False, **kwargs):
            """
            :param env: open gym environment
            :param kwargs: noise
            :param render: flag to render
            """
            self.showing_render = render
            self.current_step = 0
            if 'noise' in kwargs:
                self.noise = kwargs.get('noise')
            else:
                self.noise = 0.0

            gym.Wrapper.__init__(self, env)

        def reset(self, **kwargs):
            self.current_step = 0
            obs = self.env.reset(**kwargs) + self.noise * np.random.randn(self.env.observation_space.shape[-1])
            return obs

        def step(self, action):
            self.current_step +=1
            obs, reward, done, info = self.env.step(action)
            if self.current_step >= 200:
                done = True
            if self.showing_render:
                # Simulate and visualise the environment
                self.env.render()
            obs = obs + self.noise * np.random.randn(self.env.observation_space.shape[-1])
            # reward = reward / 10
            return obs, reward, done, info
    env = TestWrapperEnv(PendulumEnv(), render=True, noise=parameters.get('noise'))
    real_env = gym.wrappers.Monitor(env, "recordings_new", force=True)
    ############################################################
    # Setting the network parameters
    ############################################################

    network_size = 25
    # Set the priors for the anchor method:
    init_params = dict(init_stddev_1_w=np.sqrt(1),
                       init_stddev_1_b=np.sqrt(1),
                       init_stddev_2_w=1 / np.sqrt(network_size)) # normalise the data

    data_noise = parameters.get('data_noise')  # estimated aleatoric Gaussian noise standard deviation
    lambda_anchor = data_noise**2 / (np.array([init_params['init_stddev_1_w'],
                                            init_params['init_stddev_1_b'],
                                            init_params['init_stddev_2_w']]) ** 2)

    aedyna(real_env=real_env, num_epochs=num_epochs,
           steps_per_env=steps_per_epoch, algorithm='SAC', model_batch_size=model_batch_size,
           simulated_steps=simulated_steps,
           num_ensemble_models=num_ensemble_models, model_iter=model_iter, init_random_steps=init_random_steps)

project_directory = None
init_params = dict(init_stddev_1_w=np.sqrt(1),
                   init_stddev_1_b=np.sqrt(1),
                   init_stddev_2_w=1 / np.sqrt(network_size))  # normalise the data

data_noise = 0.0
lambda_anchor = data_noise ** 2 / (np.array([init_params['init_stddev_1_w'],
                                             init_params['init_stddev_1_b'],
                                             init_params['init_stddev_2_w']]) ** 2)

hyp_str_all = 'steps-epoch' + str(steps_per_epoch) + '_n-epochs' + str(num_epochs) + \
                  '_model-batchsize' + str(model_batch_size) + \
                  '_sim-steps' + str(simulated_steps) + \
                  '\nmodel-iter' + str(model_iter) + '_n-ens-models' + str(num_ensemble_models) + '_init-steps' + str(
        init_random_steps) + '/'

def new_main():
    global project_directory

    from qfb_env.qfb_nonlinear_env import QFBNLEnv

    random_seed = 123
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_name = f"AEDYNA_QFBNL"#_{datetime.strftime(datetime.now(), '%m%d%y_%H%M')}"
    project_directory = os.path.join('models', model_name)

    ppo_params = dict(
        gamma=0.99,
        n_steps=128, # 32,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lam=0.95,
        nminibatches=4,
        noptepochs=4,
        cliprange=0.2,
        cliprange_vf=None,
        policy_kwargs={'net_arch': [100, 100]}
    )

    env_kwargs = dict(rm_loc=os.path.join('..', 'metadata', 'LHC_TRM_B1.response'),
                      calibration_loc=os.path.join('..', 'metadata', 'LHC_circuit.calibration'))

    env = QFBNLEnv(**env_kwargs)
    eval_env = QFBNLEnv(**env_kwargs)

    aedyna(real_env=env, eval_env=eval_env, num_epochs=num_epochs, steps_per_env=steps_per_epoch, algorithm='PPO',
           simulated_steps=simulated_steps, num_ensemble_models=num_ensemble_models, model_iter=model_iter, model_batch_size=model_batch_size,
           init_random_steps=init_random_steps, agent_params=ppo_params)

if __name__ == '__main__':
    new_main()