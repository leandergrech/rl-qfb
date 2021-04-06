import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from normalize_env import NormalizeEnv
from replay_buffer import ReplayBuffer

tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')


class CustomTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(CustomTensorBoard, self).__init__(*args, **kwargs)
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        # Keeps track of epoch for profiling.
        self.epoch += 1
        super(CustomTensorBoard, self).on_epoch_begin(self.epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        super(CustomTensorBoard, self).on_epoch_end(self.epoch, logs)


class QModel:
    """Artificial neural net holding the state-action value function in a simple analytical form"""

    def __init__(self, obs_dim, act_dim, name, training_info=dict(), hidden_sizes=(100, 100), activation=tf.nn.tanh,
                 kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.01, 0.01), early_stopping_patience=2,
                 save_frequency=1000):

        self.save_frequency = save_frequency
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=early_stopping_patience)

        if name is not None:
            self.__name__ = name
            #print(self.__name__)

        self.polyak = training_info.get('polyak', 0.999)
        self.discount = training_info.get('discount', 0.999)
        self.steps_per_batch = training_info.get('steps_per_batch', 10)
        self.batch_size = training_info.get('batch_size', 1)
        self.learning_rate = training_info.get('learning_rate', 1e-3)
        self.epochs = training_info.get('epochs')

        self.init = True

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Define the network inputs (state-action)
        inputs_state = Input(shape=(self.obs_dim,), name="state_input")
        inputs_action = Input(shape=(self.act_dim,), name="action_input")

        # create a shared network for the variables
        h = inputs_state
        for hidden_dim in self.hidden_sizes:
            h = self.fc(h, hidden_dim, kernel_initializer=kernel_initializer, activation=self.activation)

        # Output - state-value function, where the reward is assumed to be negative
        V = -self.fc(h, 1, activation=tf.nn.leaky_relu,
                     kernel_initializer=kernel_initializer, name='V')
        # Output - for the matrix L
        l = self.fc(h, (self.act_dim * (self.act_dim + 1) / 2),
                    kernel_initializer=kernel_initializer, name='l')
        # Output - policy pi
        mu = self.fc(h, self.act_dim, kernel_initializer=kernel_initializer, name='mu')
        self.value_model = Model([inputs_state], V, name='value_model')
        self.action_model = Model([inputs_state], mu, name='action_model')

        pivot = 0
        rows = []
        for idx in range(self.act_dim):
            count = self.act_dim - idx
            diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
            non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
            row = tf.pad(tensor=tf.concat((diag_elem, non_diag_elems), 1), paddings=((0, 0), (idx, 0)))
            rows.append(row)
            pivot += count

        L = tf.transpose(a=tf.stack(rows, axis=1), perm=(0, 2, 1))
        P = tf.matmul(L, tf.transpose(a=L, perm=(0, 2, 1)))
        tmp = tf.expand_dims(inputs_action - mu, -1)

        # The advantage function
        A = -tf.multiply(tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]),
                                   tf.matmul(P, tmp)),
                         tf.constant(0.5, dtype=tf.float64))
        A = tf.reshape(A, (-1, 1))

        # The state-action-value function
        Q = tf.add(A, V)

        # We use a customized way to train the model:
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.q_model = self.CustomModel(inputs=[inputs_state, inputs_action], outputs=Q, mother_class=self)
        self.q_model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])

    def fc(self, x, hidden_size, activation=tf.nn.tanh,
           kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.01, 0.01),
           name=None):
        layer = Dense(hidden_size, activation=activation,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=None,
                                   bias_initializer=tf.compat.v1.constant_initializer(0.0), name=name)
        return layer(x)

    def get_action(self, state):
        return self.action_model.predict(np.array(state))

    def get_value_estimate(self, state):
        return self.value_model.predict(np.array(state))

    def polyak_average(self, weights):
        weights_old = self.get_weights()
        weights_new = [self.polyak * weights_old[i] + (1 - self.polyak) * weights[i] for i in range(len(weights))]
        self.q_model.set_weights(weights=weights_new)

    def get_weights(self):
        return self.q_model.get_weights()

    def save_model(self, filename):
        try:
            self.q_model.save_weights(filepath=filename, save_format='h5')
        except Exception as e:
            print(f'Saving failed: {e}')

    def load_model(self, directory):
        try:
            self.q_model.load_weights(filepath=os.path.join(directory, f'{self.__name__}.h5'))
        except Exception as e:
            print(f'Loading failed: {e}')

    def set_target_models(self, q_target_1, q_target_2):
        self.q_target_first = q_target_1
        self.q_target_second = q_target_2

    def batch_training_step(self, batch):
        # batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        # Here we decide how often to iterate over the data
        dataset = tf.data.Dataset.from_tensor_slices(batch)  # .repeat(1).shuffle(buffer_size=10000)
        train_dataset = dataset.batch(self.steps_per_batch)

        '''
        put the following in a range(epochs) and apply polyak per epoch'''
        hist = self.q_model.fit(train_dataset,
                                verbose=0,
                                shuffle=True,
                                # callbacks=self.tensorboard_callback,
                                # batch_size=self.training_params['batch_size'],
                                epochs=self.epochs)
        self.q_target_first.polyak_average(self.q_model.get_weights())#, self.polyak, name=self.q_model.__name__)

        # if int(self.ckpt.step) % self.save_frequency == 0:
        #     save_path = self.manager.save()
        #     print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        #     save_path_target = self.q_target_first.manager.save()
        #     print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_target))
        # self.ckpt.step.assign_add(1)
        return_value = hist.history['loss']

        return return_value

    class CustomModel(Model):

        def __init__(self, *args, **kwargs):
            self.mother_class = kwargs.pop('mother_class')
            self.__name__ = self.mother_class.__name__

            super().__init__(*args, **kwargs)

        def train_step(self, batch):
            self.discount = self.mother_class.discount
            self.polyak = self.mother_class.polyak

            v_1 = self.mother_class.q_target_first.value_model(batch['obs2'])  # , training=False)
            v_2 = self.mother_class.q_target_second.value_model(batch['obs2'])  # , training=False)
            v = tf.squeeze(tf.where(tf.math.less(v_1, v_2), v_1, v_2))

            y_target = tf.add(tf.multiply(tf.math.scalar_mul(self.discount, v),
                                          tf.add(tf.constant(1, dtype=tf.float64),
                                                 tf.math.scalar_mul(-1, batch['done']))), batch['rews'])

            # Double Q implementation
            # a_1 = self.mother_class.q_target_first.action_model(batch['obs2'])
            # q_2 = self.mother_class.q_target_second.q_model([batch['obs2'], a_1])
            #
            # y_target = tf.add(tf.multiply(tf.math.scalar_mul(self.discount, q_2),
            #                               tf.add(tf.constant(1, dtype=tf.float64),
            #                                      tf.math.scalar_mul(-1, batch['done']))), batch['rews'])

            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred = self([batch['obs1'], batch['acts']], training=True)
                # Compute the loss value for this minibatch.
                loss = self.compiled_loss(y_target, y_pred)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            # Compute gradients
            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update the metrics.
            # Metrics are configured in `compile()`.
            self.compiled_metrics.update_state(y_target, y_pred)
            # Apply weights to target network
            return {m.name: m.result() for m in self.metrics}


class NAF2(object):
    MAX_STEPS = 1000

    def __init__(self, env, buffer_size=int(1e6), train_every=1,
                 training_info={'polyak':0.999,
                                'batch_size':100,
                                'steps_per_batch':10,
                                'epochs':1,
                                'learning_rate':1e-3,
                                'discount':0.9999},
                 eval_info=dict(),
                 nafnet_info={'hidden_sizes': [100, 100],
                              'activation': tf.nn.tanh,
                              'kernel_initializer': tf.random_normal_initializer(0, 0.05)},
                 save_frequency=1000, log_frequency=100, directory=None,
                 tb_log=None, q_smoothing_sigma=0.02, q_smoothing_clip=0.05, soft_init=False,
                 noise_fn=None):
        """
        :param env: open gym environment to be solved
        :dict training_info: dictionary containing info for the training of the network
        :param noise_info: dict with noise function for decay of gaussian noise
        :param save_frequency: frequency to save the weights of the network
        :param directory: directory were weights are saved
        :param is_continued: continue a training, otherwise given directory deleted if existing
        :param q_smoothing_clip: add small noise on actions to smooth the training
        :param q_smoothing: add small noise on actions to smooth the training
        :param nafnet_info: keywords to handle the network and training
        """

        self.env = NormalizeEnv(env)
        self.act_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]

        self.train_every = train_every
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency

        self.eval_env = eval_info.get('eval_env', None) # If None, no evaluation is done during training
        self.eval_frequency = eval_info.get('frequency', 1000)
        self.eval_nb_eps = eval_info.get('nb_episodes', 10)
        self.eval_max_steps = eval_info.get('max_ep_steps', 100)

        self.q_smoothing_sigma = q_smoothing_sigma
        self.q_smoothing_clip = q_smoothing_clip

        self.losses_q1 = []
        self.losses_q2 = []
        self.vs = []
        self.vs2 = []

        if noise_fn is None:
            self.noise_function = lambda action, nr: action + np.random.randn(self.act_dim) * 1 / (nr + 1)
        else:
            self.noise_function = noise_fn

        self.idx_episode = None

        self.training_info = training_info
        self.learning_rate = training_info.get('learning_rate', 1e-3)
        self.batch_size = training_info.get('batch_size', 100)

        self.directory = directory
        self.tb_writer = tf.summary.create_file_writer(tb_log).set_as_default()

        if not soft_init:
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                assert False, 'Directory passed already exists'

        # Create working models
        self.q_main_model_1 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                     name='q_main_model_1',
                                     training_info=training_info,
                                     save_frequency=self.save_frequency,
                                     **nafnet_info)
        self.q_main_model_2 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                     name='q_main_model_2',
                                     training_info=training_info,
                                     save_frequency=self.save_frequency,
                                     **nafnet_info)

        # Create target models
        self.q_target_model_1 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                       name='q_target_model_1',
                                       training_info=training_info,
                                       **nafnet_info)
        self.q_target_model_2 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                       name='q_target_model_2',
                                       training_info=training_info,
                                       **nafnet_info)

        # Working and target models have the same initial values
        self.q_target_model_1.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())
        self.q_target_model_2.q_model.set_weights(weights=self.q_main_model_2.q_model.get_weights())

        # Set the target models of each respective working model
        self.q_main_model_1.set_target_models(self.q_target_model_1, self.q_target_model_2)
        self.q_main_model_2.set_target_models(self.q_target_model_2, self.q_target_model_1)

        self.it = 0
        self.memory = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

    # deterministic predict
    def predict(self, state):
        return self._predict(self.q_target_model_1, state, False)

    def _predict(self, model, state, is_train):
        if is_train and self.memory.size < self.warm_up_steps:
            # action = np.random.uniform(-1, 1, self.action_size)
            action = self.env.action_space.sample()
            return np.array(action)
        elif is_train:
            action = model.get_action([state])
            action = self.noise_function(action, self.idx_episode)
            return action
        else:
            action = model.get_action([state])
            return action

        # if is_train and self.memory.size < self.warm_up_steps:
        #     # action = np.random.uniform(-1, 1, self.action_size)
        #     action = self.env.action_space.sample()
        #     return np.array(action)
        #
        # # Add small noise on the controller
        # elif is_train:
        #     action = self.noise_function(np.squeeze(model.get_action([state])),self.idx_episode)
        #     if self.q_smoothing_clip is None:
        #         return_value = np.clip(action, -1, 1)
        #     else:
        #         return_value = np.clip(action + np.clip(self.q_smoothing_sigma * np.random.randn(self.act_dim),
        #                                                 -self.q_smoothing_clip, self.q_smoothing_clip),
        #                                -1, 1)
        #     return return_value
        # else:
        #     action = model.get_action([state])
        #     return action

    def training(self, nb_steps=int(1e5), max_ep_steps=100, warm_up_steps=100, initial_episode_length=5):
        self.warm_up_steps = warm_up_steps
        self.initial_episode_length = initial_episode_length

        self.max_ep_steps = max_ep_steps
        self.nb_steps = nb_steps

        self._training()

    @property
    def _warmup_done(self):
        return self.memory.size > self.warm_up_steps

    def _training(self):
        self.idx_episode = 0
        ep_step = 0
        o = self.env.reset()
        ep_len = []
        pbar = tqdm(total=self.nb_steps)
        for t in range(0, self.nb_steps):
            self.it = t
            # 1. predict
            a = np.squeeze(self._predict(model=self.q_main_model_1, state=o, is_train=True))

            if self.q_smoothing_clip is None:
                a = np.clip(a, -1, 1)
            else:
                a = np.clip(a + np.clip(self.q_smoothing_sigma * np.random.randn(self.act_dim),
                                                        -self.q_smoothing_clip, self.q_smoothing_clip),
                                       -1, 1)

            o2, r, d, _ = self.env.step(a)
            ep_step += 1

            self.memory.store(o, a, r, o2, d)

            o = o2
            d = False if ep_step == self.max_ep_steps else d

            # Initial reset allows us to collect samples close to the initial state, without getting crazy states
            if t > 0 and t % self.initial_episode_length == 0 and not self._warmup_done:
                o = self.env.reset()
                ep_step = 0
            elif (ep_step % self.max_ep_steps == 0 or d) and self._warmup_done:
                o = self.env.reset()
                self.idx_episode += 1
                ep_len.append(ep_step)
                ep_step = 0


            # 2. train & log
            if self._warmup_done:
                if t % self.train_every == 0:
                    loss_q1 = self.q_main_model_1. \
                        batch_training_step(self.memory.sample_batch(self.batch_size))[-1]
                    loss_q2 = self.q_main_model_2. \
                        batch_training_step(self.memory.sample_batch(self.batch_size))[-1]

                    self.losses_q1.append(loss_q1)
                    self.losses_q2.append(loss_q2)

                if t % self.eval_frequency == 0:
                    self.evaluate()

                if t % self.log_frequency == 0:
                    lookback = self.log_frequency//self.train_every
                    tf.summary.scalar('loss/q_main_model_1', data=np.mean(self.losses_q1[-lookback:]), step=self.it)
                    tf.summary.scalar('loss/q_main_model_2', data=np.mean(self.losses_q2[-lookback:]), step=self.it)
                    if self.idx_episode > 0:
                        tf.summary.scalar('training/average_episode_length', data=np.mean(ep_len[-lookback:]), step=self.it)
                    tf.summary.scalar('info/episode_idx', data=self.idx_episode, step=self.it)
                    tf.summary.flush()

                if t % self.save_frequency == 0:
                    self.save_checkpoint()

            if t % int(self.nb_steps/100) == 0:
                pbar.update(int(self.nb_steps/100))

    # def _training(self, is_train=True):
    #     pbar = tqdm(total=self.max_episodes * self.max_ep_steps)
    #     for index in range(0, self.max_episodes):
    #         self.idx_episode = index
    #
    #         o = self.env.reset()
    #         for t in range(0, self.max_ep_steps):
    #             # 1. predict
    #             a = np.squeeze(self._predict(self.q_main_model_1, o, is_train))
    #             o2, r, d, _ = self.env.step(a)
    #
    #             if is_train:
    #                 self.memory.store(o, a, r, o2, d)
    #
    #             o = o2
    #             d = False if t == self.max_ep_steps - 1 else d
    #
    #             if t > 0 and t % self.initial_episode_length == 0 and self.memory.size <= self.warm_up_steps:
    #                 o = self.env.reset()
    #                 # print('Initial reset at ', t)
    #
    #             # 2. train
    #             if self.it % self.train_every == 0 and is_train and self.memory.size > self.warm_up_steps:
    #                 loss_q1 = self.q_main_model_1.\
    #                     batch_training_step(self.memory.sample_batch(self.batch_size))[-1]
    #                 loss_q2 = self.q_main_model_2.\
    #                     batch_training_step(self.memory.sample_batch(self.batch_size))[-1]
    #
    #                 self.losses_q1.append(loss_q1)
    #                 self.losses_q2.append(loss_q2)
    #
    #             if self.it > 0 and self.memory.size > self.warm_up_steps:
    #                 if self.it % self.save_frequency == 0:
    #                     #print(f'-> Saving checkpoint on step {self.it}')
    #                     self.save_checkpoint()
    #
    #                 if self.it % self.eval_frequency == 0:
    #                     #print(f'-> Evaluating on step {self.it}')
    #                     self.evaluate()
    #
    #                 if self.it % self.log_frequency == 0:
    #                     #print(f'-> Logging on step {self.it}')
    #                     lookback = self.log_frequency//self.train_every
    #                     tf.summary.scalar('loss/q_main_model_1', data=np.mean(self.losses_q1[-lookback:]), step=self.it)
    #                     tf.summary.scalar('loss/q_main_model_2', data=np.mean(self.losses_q2[-lookback:]), step=self.it)
    #                     tf.summary.flush()
    #
    #             self.it += 1
    #             pbar.update(1)
    #             if d:
    #                 break
    #     pbar.close()

    def save_checkpoint(self, save_buffer=True, save_main_models=True, save_target_models=True):
        if not any((save_buffer, save_main_models, save_target_models)):
            print('Nothing is marked for saving')
            return

        number = str(self.it).zfill(4)
        par_dir = os.path.join(self.directory, f'step_{number}')
        os.makedirs(par_dir)
        if save_buffer:
            self.memory.save_to_pkl(name=f'buffer_data.pkl', directory=par_dir)

        if save_main_models:
            # main_models_dir = os.path.join(par_dir, 'main_models')
            for model in (self.q_main_model_1, self.q_main_model_2):
                model.save_model(os.path.join(par_dir, f'{model.__name__}.h5'))
        if save_target_models:
            # target_models_dir = os.path.join(par_dir, 'target_models')
            for model in (self.q_target_model_1, self.q_target_model_2):
                model.save_model(os.path.join(par_dir, f'{model.__name__}.h5'))

    def load_checkpoint(self, model_dir, chkpt_step, load_buffer=True):
        self.directory = model_dir
        number = str(chkpt_step).zfill(4)
        par_dir = os.path.join(model_dir, f'step_{number}')
        if load_buffer:
            self.memory.read_from_pkl(name=f'buffer_data.pkl',
                                      directory=par_dir)
        self.q_main_model_1.load_model(par_dir)
        self.q_main_model_2.load_model(par_dir)
        self.q_target_model_1.load_model(par_dir)
        self.q_target_model_2.load_model(par_dir)

    def evaluate(self):
        if self.eval_env is None:
            return

        env = self.eval_env
        ret = []
        for ep in range(self.eval_nb_eps):
            o = env.reset()
            ret.append(0.0)
            for step in range(self.eval_max_steps):
                a = self._predict(self.q_target_model_1, o, False).squeeze()
                o, r, d, _ = env.step(a)
                ret[-1] += r
                if d:
                    break
        mean_return = np.mean(ret)
        tf.summary.scalar('training/episode_return', data=mean_return, step=self.it)

    def visualize(self):
        state = np.zeros(self.env.observation_space.shape)

        delta = 0.05
        theta = np.arange(-1, 1, delta)
        theta_dot = np.arange(-1, 1, delta)
        X, Y = np.meshgrid(theta, theta_dot)

        Nr = 1
        Nc = 2
        fig, axs = plt.subplots(Nr, Nc)
        fig.subplots_adjust(hspace=0.3)

        rewards = np.zeros(X.shape)
        actions = np.zeros(X.shape)
        for i1 in range(len(theta)):
            for j1 in range(len(theta_dot)):
                state[0] = np.sin(theta[i1])
                state[1] = np.cos(theta[i1])
                state[2] = theta_dot[j1]

            rewards[i1, j1] = self.q_target_model_1.get_value_estimate([state])
            actions[i1, j1] = self.q_target_model_1.get_action([state])

        axs[0].contour(X, Y, rewards, alpha=1)
        axs[0].set_title('Value estimate')

        axs[1].contour(X, Y, actions, alpha=1)
        axs[1].set_title('Policy estimate')

        fig.show()