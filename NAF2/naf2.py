import os
import pickle as pkl
import shutil

import numpy as np
import matplotlib.pyplot as plt

import gym
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from ..envs.normalize_env import NormalizeEnv
from ..ReplayBuffer.replay_buffer import ReplayBuffer

tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')


class QModel:
    """Artificial neural net holding the state-action value function in a simple analytical form"""

    def __init__(self, obs_dim, act_dim, training_info=dict(), hidden_sizes=(100, 100),
                 kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.01, 0.01), early_stopping_patience=2,
                 save_frequency=500, name=None, directory=None):
        self.tensorboard_callback = TensorBoard(log_dir=os.path.join(directory, 'logs'), histogram_freq=1)

        self.directory = directory
        self.save_frequency = save_frequency
        self.hidden_sizes = hidden_sizes

        self.callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=early_stopping_patience)

        if name is not None:
            self.__name__ = name
            print(self.__name__)

        self.polyak = training_info.get('polyak', 0.999)
        self.discount = training_info.get('discount', 0.999)
        self.steps_per_batch = training_info.get('steps_per_batch', 1)
        self.batch_size = training_info.get('batch_size', 1)
        self.learning_rate = training_info.get('learning_rate', 1e-3)
        self.training_params = training_info

        self.kernel_initializer = kernel_initializer

        self.init = True

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Define the network inputs (state-action)
        inputs_state = Input(shape=(self.obs_dim,), name="state_input")
        inputs_action = Input(shape=(self.act_dim,), name="action_input")

        # create a shared network for the variables
        h = inputs_state
        for hidden_dim in self.hidden_sizes:
            h = self.fc(h, hidden_dim, kernel_initializer=self.kernel_initializer)

        # Output - state-value function, where the reward is assumed to be negative
        V = tf.scalar_mul(-1, self.fc(h, 1, activation=tf.nn.leaky_relu,
                                      kernel_initializer=self.kernel_initializer, name='V'))
        # Output - for the matrix L
        l = self.fc(h, (self.act_dim * (self.act_dim + 1) / 2),
                    kernel_initializer=self.kernel_initializer, name='l')
        # Output - policy pi
        mu = self.fc(h, self.act_dim, kernel_initializer=self.kernel_initializer, name='mu')
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


        self.storage_management()

    def storage_management(self):
        checkpoint_dir = self.directory + self.__name__ + "/"
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.q_model)
        self.manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

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

    def save_model(self, directory):
        try:
            self.q_model.save(filepath=directory, overwrite=True)
        except:
            print('Saving failed')

    def set_target_models(self, q_target_1, q_target_2):
        self.q_target_first = q_target_1
        self.q_target_second = q_target_2

    def set_training_parameters(self, **kwargs):
        # Filter kwargs for keras
        self.polyak = kwargs.get('polyak', 0.999)
        self.discount = kwargs.get('discount', 0.999)
        self.steps_per_batch = kwargs.get('steps_per_batch', 1)
        self.batch_size = kwargs.get('batch_size', 1)

        self.callback = self.CustomCallback(patience=0)
        self.callback.q_target = self.q_target_first

        self.training_params = kwargs

    def batch_training_step(self, batch):
        # batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        # Here we decide how often to iterate over the data
        dataset = tf.data.Dataset.from_tensor_slices(batch)  # .repeat(1).shuffle(buffer_size=10000)
        train_dataset = dataset.batch(self.steps_per_batch)

        hist = self.q_model.fit(train_dataset,
                                verbose=0,
                                shuffle=True,
                                **self.training_params)
        self.q_target_first.polyak_average(self.q_model.get_weights(), self.polyak, name=self.q_model.__name__)

        if int(self.ckpt.step) % self.save_frequency == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            save_path_target = self.q_target_first.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_target))
        self.ckpt.step.assign_add(1)
        return_value = hist.history['loss']

        return return_value

    def create_buffers(self, buffer=None):
        if buffer is None:
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, max_size=int(1e6))
            try:
                files = []
                directory = self.directory + 'data/'
                for f in os.listdir(directory):
                    if 'buffer_data' in f and 'pkl' in f:
                        files.append(f)
                files.sort()
                self.replay_buffer.read_from_pkl(name=files[-1], directory=directory)
                print('Buffer data loaded for ' + self.__name__, files[-1])
            except:
                print('Buffer data empty for ' + self.__name__, files)
        else:
            self.replay_buffer = buffer

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

    def __init__(self, env, buffer_size=int(1e6), train_every=1, training_info=dict(), pre_tune=None,
                 save_frequency=1000, eval_frequency=1000, directory=None, tensorboard_log=None,
                 q_smoothing_sigma=0.02, q_smoothing_clip=0.05, **nafnet_kwargs):
        """
        :param env: open gym environment to be solved
        :dict training_info: dictionary containing info for the training of the network
        :tuple pre_tune: list of tuples (state action reward next state done)
        :param noise_info: dict with noise function for decay of gaussian noise
        :param save_frequency: frequency to save the weights of the network
        :param directory: directory were weights are saved
        :param is_continued: continue a training, otherwise given directory deleted if existing
        :param q_smoothing_clip: add small noise on actions to smooth the training
        :param q_smoothing: add small noise on actions to smooth the training
        :param nafnet_kwargs: keywords to handle the network and training
        """

        self.env = NormalizeEnv(env)
        self.act_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]

        self.train_every = train_every
        self.save_frequency = save_frequency
        self.eval_frequency = eval_frequency

        self.q_smoothing_sigma = q_smoothing_sigma
        self.q_smoothing_clip = q_smoothing_clip

        self.losses_q1 = []
        self.losses_q2 = []
        self.vs = []
        self.vs2 = []

        self.pre_tune = pre_tune

        self.noise_function = lambda action, nr: action + np.random.randn(self.act_dim) * 1 / (nr + 1)

        self.idx_episode = None

        self.training_info = training_info
        self.learning_rate = training_info.get('learning_rate', 1e-3)
        self.batch_size = training_info.get('batch_size', 128)
        self.steps_per_batch = training_info.get('steps_per_batch', 10)

        self.directory = directory

        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            assert False, 'Directory passed already exists'

        if not os.path.exists(tensorboard_log):
            os.makedirs(tensorboard_log)

        # Create working models
        self.q_main_model_1 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                     name='q_main_model_1',
                                     directory=self.directory,
                                     save_frequency=self.save_frequency,
                                     training_info=training_info,
                                     **nafnet_kwargs)
        self.q_main_model_2 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                     name='q_main_model_2',
                                     directory=self.directory,
                                     save_frequency=self.save_frequency,
                                     training_info=training_info,
                                     **nafnet_kwargs)

        # Create target models
        self.q_target_model_1 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                       name='q_target_model_1',
                                       directory=self.directory,
                                       training_info=training_info,
                                       **nafnet_kwargs)
        self.q_target_model_2 = QModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                       name='q_target_model_2',
                                       directory=self.directory,
                                       training_info=training_info,
                                       **nafnet_kwargs)

        # Working and target models have the same initial values
        self.q_target_model_1.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())
        self.q_target_model_2.q_model.set_weights(weights=self.q_main_model_2.q_model.get_weights())

        # Set the target models of each respective working model
        self.q_main_model_1.set_target_models(self.q_target_model_1, self.q_target_model_2)
        self.q_main_model_2.set_target_models(self.q_target_model_2, self.q_target_model_1)

        self.updates = 0

        self.memory = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

    def predict(self, model, state, is_train):
        if is_train and model.replay_buffer.size < self.warm_up_steps:
            # action = np.random.uniform(-1, 1, self.action_size)
            action = self.env.action_space.sample()
            return np.array(action)

        # Add small noise on the controller
        elif is_train:
            action = self.noise_function(np.squeeze(model.get_action([state])),self.idx_episode)
            if self.q_smoothing_clip is None:
                return_value = np.clip(action, -1, 1)
            else:
                return_value = np.clip(action + np.clip(self.q_smoothing_sigma * np.random.randn(self.act_dim),
                                                        -self.q_smoothing_clip, self.q_smoothing_clip),
                                       -1, 1)
            return return_value
        else:
            action = model.get_action([state])
            return action

    def training(self, warm_up_steps=0, initial_episode_length=5, **kwargs):
        self.warm_up_steps = warm_up_steps
        self.initial_episode_length = initial_episode_length

        if 'max_episodes' in kwargs:
            self.max_episodes = kwargs.get('max_episodes')
        if 'max_steps' in kwargs:
            self.max_steps = kwargs.get('max_steps')

        self._training(is_train=True)

    def _training(self, is_train=True):
        for index in tqdm(range(0, self.max_episodes)):
            self.idx_episode = index

            o = self.env.reset()
            for t in range(0, self.max_steps):
                # 1. predict
                a = np.squeeze(self.predict(self.q_main_model_1, o, is_train))
                o2, r, d, _ = self.env.step(a)

                if is_train:
                    self.memory.store(o, a, r, o2, d)

                o = o2
                d = False if t == self.max_steps - 1 else d

                if t > 0 and t % self.initial_episode_length == 0 and self.memory.size <= self.warm_up_steps:
                    o = self.env.reset()
                    print('Initial reset at ', t)

                # 2. train
                if t % self.train_every == 0:
                    if is_train and self.memory.size > self.warm_up_steps:
                        self.updates += 1
                        loss_q1 = self.q_main_model_1.\
                            batch_training_step(self.memory.sample_batch(self.batch_size))[-1]
                        loss_q2 = self.q_main_model_2.\
                            batch_training_step(self.memory.sample_batch(self.batch_size))[-1]

                        self.losses_q1.append(loss_q1)
                        self.losses_q2.append(loss_q2)

                if t % self.save_frequency == 0:
                    self.save_checkpoint()

                if d:
                    break

    def save_checkpoint(self, save_buffer=True, save_main_models=True, save_target_models=True):
        if not any((save_buffer, save_main_models, save_target_models)):
            print('Nothing is marked for saving')
            return

        print(f'-> Saving checkpoint on step {self.updates}')
        number = str(self.updates).zfill(4)
        if save_buffer:
            print('\t`-> Saving buffer')
            buffer_dir = os.path.join(self.directory, 'buffer_data')
            self.memory.save_to_pkl(name=f'buffer_data_{number}.pkl', directory=buffer_dir)
        if save_main_models:
            print('\t1-> Saving main models')
            main_models_dir = os.path.join(self.directory, 'main_models')
            for model in (self.q_main_model_1, self.q_main_model_2):
                model.save_model(os.path.join(main_models_dir, model.__name__))
        if save_target_models:
            print('\t1-> Saving target models')
            main_models_dir = os.path.join(self.directory, 'target_models')
            for model in (self.q_target_model_1, self.q_target_model_2):
                model.save_model(os.path.join(main_models_dir, model.__name__))

    def update_q(self, model, **kwargs):

        loss = model.batch_training_step(self.memory.sample_batch(self.batch_size), **kwargs)[-1]

        return loss


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
        axs[0].set_title('Policy estimate')

        fig.show()