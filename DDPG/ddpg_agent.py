import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input,concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from stable_baselines.common.buffers import ReplayBuffer
from tqdm import tqdm

# from ReplayBuffer.my_buffer import MyReplayBuffer

class DDPGAgent:
    def __init__(self, obs_dimension, act_dimension):
        # Parameters
        self._obs_dimension = obs_dimension
        self._act_dimension = act_dimension
        self.ACTOR_LR = 5e-4
        self.CRITIC_LR = 1e-3
        self.GAMMA = 0.99
        self.TAU = 0.001
        self.BUFFER_SIZE = int(5e5)
        self.BATCH_SIZE = 64
        self.ACT_NOISE_SCALE = 0.1
        self.ACT_LIMIT = 1.0
        self.REWARD_NORM = 100
        self.EPS = np.finfo(np.float32).eps.item()

        self._hidden_actor = [600, 300]
        self._hidden_critic = [600, 300]

        self.returns_mean = None
        self.returns_std = None

        self.optimizer_actor = Adam(learning_rate=self.ACTOR_LR)
        self.optimizer_critic = Adam(learning_rate=self.CRITIC_LR)
        self.actor = self._generate_actor()
        self.actor_target = self._generate_actor()
        self.critic = self._generate_critic()
        self.critic_target = self._generate_critic()

        self.memory = ReplayBuffer(self.BUFFER_SIZE)#, obs_dimension, act_dimension)

        self.dummy_Q_target_prediction_input = np.zeros((self.BATCH_SIZE, 1))
        self.dummy_dones_input = np.zeros((self.BATCH_SIZE, 1))

    """
    ACTOR methods
    """
    def _generate_actor(self):
        """
        Generates the actor network
        Prints summary of network model
        :return: Tuple containing references to the model, weights, and input layer
        """
        model = Sequential()
        model.add(Dense(self._hidden_actor[0], input_shape=(self._obs_dimension,), activation='relu'))
        for h in self._hidden_actor[1:]:
            model.add(BatchNormalization())
            model.add(Dense(h, activation='relu'))
        model.add(Dense(self._act_dimension, activation='tanh'))
        model.compile(optimizer=self.optimizer_actor, loss=self._ddpg_actor_loss)
        # model.summary()

        return model

    def _ddpg_actor_loss(self, y_true, y_pred):
        # y_true is Q_prediction = Q_critic_predicted(s, a_actor,_predicted)
        return -tf.reduce_mean(y_true)

    def get_action(self, states, noise=None):
        """
        Returns the action (=prediction of local actor) given a state

        :param states:
        :param noise:
        :return:
        """
        if noise is None:
            noise = self.ACT_NOISE_SCALE
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        action = self.actor.predict_on_batch(states)
        if noise != 0:
            action += noise * np.random.randn(self._act_dimension)
            action = np.clip(action, -self.ACT_LIMIT, self.ACT_LIMIT)
        return action

    def get_target_action(self, states):
        return self.actor_target(states)

    @tf.function
    def train_actor(self, states, actions):
        with tf.GradientTape() as tape:
            mu = self.actor(states)
            q_pred = self.critic([states, mu])#, self.dummy_Q_target_prediction_input, self.dummy_dones_input])
        mu_grad = tape.gradient(q_pred, self.actor.trainable_weights)
        mu_grad = [tf.divide(tf.multiply(t, -1.0), self.BATCH_SIZE) for t in mu_grad]
        self.optimizer_actor.apply_gradients(zip(mu_grad, self.actor.trainable_variables))

    """
    CRITIC methods
    """
    def _generate_critic(self):
        # Inputs to the network. Most are used by the loss function, not for FF
        state_input = Input(shape=self._obs_dimension, name='state_input')
        action_input = Input(shape=self._act_dimension, name='action_input')
        # q_target_prediction_input = Input(shape=(1,), name='Q_target_prediction_input')
        # dones_input = Input(shape=(1,), name='dones_input')

        # Define network structure
        state_action_pair = concatenate([state_input, action_input])
        dense = Dense(self._hidden_critic[0], activation='relu')(state_action_pair)
        for h in self._hidden_critic[1:]:
            dense = BatchNormalization()(dense)
            dense = Dense(h, activation='relu')(dense)
        out = Dense(1, activation='linear')(dense)

        # Create model
        model = Model(inputs=[state_input, action_input], outputs=out)
        model.compile(optimizer=self.optimizer_critic, loss='mse')
                      # loss=self._ddpg_critic_loss(q_target_prediction_input, dones_input))
        # model.summary()

        return model

    # @tf.function
    # def _ddpg_critic_loss(self, q_target_prediction_input, dones_input):
    #     def loss(y_true, y_pred):
    #         # y_true = rewards ; y_pred = Q
    #         target_q = y_true + (self.GAMMA * q_target_prediction_input *
    #                              (1 - dones_input))
    #         err = mse(target_q, y_pred)
    #         return err
    #     return loss

    def train_critic(self, o, otp1, a, r, d):
        """
        Train critic using a trajectory batch from memory
        Loss of the critic requires the predicted Q(s,a) which is the prediction
        of the critic
        """
        q_target = self.get_target_q(otp1, d, r)
        loss = self.critic.train_on_batch([o, a], q_target)
        return loss

    def get_q(self, states, actions):
        return self.critic([states, actions])#, self.dummy_Q_target_prediction_input, self.dummy_dones_input])

    def get_target_q(self, next_states, dones, rewards):
        qtp1_pred = tf.reshape(self.critic_target([next_states, self.actor(next_states)]), (-1,))

        returns =  (rewards + self.GAMMA * qtp1_pred) * (1 - dones)

        returns_mean = tf.math.reduce_mean(returns)
        returns_std = tf.math.reduce_std(returns)
        if self.returns_mean is None:
          self.returns_mean = returns_mean
          self.returns_std = returns_std
        else:
          self.returns_mean = tf.math.reduce_mean([self.returns_mean, returns_mean])
          self.returns_std = tf.math.reduce_mean([self.returns_std, returns_std])

        return tf.reshape((returns - self.returns_mean) / (self.returns_std + self.EPS), (-1, 1))


        # return self.critic_target.predict_on_batch([states, actions,
        #             self.dummy_Q_target_prediction_input, self.dummy_dones_input])

    """
    AGENT methods
    """
    def _soft_update_actor_and_critic(self):
        """
        Polyak averaging of both networks using self.TAU to control rate
        """
        weights_critic_local = np.array(self.critic.get_weights())
        weights_critic_target = np.array(self.critic_target.get_weights())
        self.critic_target.set_weights(self.TAU * weights_critic_local +
                                       (1.0 - self.TAU * weights_critic_target))

        weights_actor_local = np.array(self.actor.get_weights())
        weights_actor_target = np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(self.TAU * weights_actor_local +
                                      (1.0 - self.TAU) * weights_actor_target)

    def store(self, o, a, r, otp1, d):
        self.memory.add(o, a, r, otp1, d)

    def train(self):
        """
        Trains the networks of the agent (local actor and critic) and Polyak
        averages their target
        """
        o, a, r, otp1, d = self.memory.sample(batch_size=self.BATCH_SIZE)
        a = tf.clip_by_value(a, -self.ACT_LIMIT, self.ACT_LIMIT)
        loss_critic = self.train_critic(o, otp1, a, r, d)
        self.train_actor(o, a)
        self._soft_update_actor_and_critic()
        return loss_critic



def train_lunar_lander():
    GAME = 'LunarLanderContinuous-v2'
    EPOCHS = 400
    MAX_EPISODE_LENGTH = 3000
    START_STEPS = 2000
    RENDER_EVERY = 1

    env = gym.make(GAME)
    agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0])

    o, r, d, ep_r, ep_l, ep_cnt = env.reset(), 0, False, [0.0], 0, 0
    loss_critic, loss_actor = [], []
    total_steps = MAX_EPISODE_LENGTH * EPOCHS

    import matplotlib.pyplot as plt
    plt.ion()

    # fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig, (ax1, ax2) = plt.subplots(2)
    ret_line, = ax1.plot([], [])
    lc_line, = ax2.plot([], [])
    # la_line, = ax3.plot([], [])

    ax1.set_ylabel('Return')
    ax2.set_ylabel('Loss Critic')
    # ax3.set_ylabel('Loss Actor')
    for ax in fig.get_axes():
        ax.set_xlabel('Episode')


    # plt.pause(1e-9)
    # plt.show(block=False)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if ep_cnt % RENDER_EVERY == 0:
            env.render()
        # get_action, at the start randomly later by policy+noise
        if t > START_STEPS:
            a = agent.get_action(o)
            a = np.squeeze(a)
        else:
            a = env.action_space.sample()

        # Step the env
        otp1, r, d, _ = env.step(a)
        ep_r[-1] += r
        ep_l += 1

        # Ignore done signal if it comes from hitting the time horizon
        if ep_l == MAX_EPISODE_LENGTH:
            d = False

        agent.store(o, a, r, otp1, d)

        o = otp1

        if d or ep_l == MAX_EPISODE_LENGTH:
            ep_cnt += 1
            # if ep_cnt % RENDER_EVERY == 0:
            if True:
                print(f"Ep: {len(ep_r) - 1}, Len: {ep_l}, RetAverage: {np.mean(ep_r[-100:])}")

            ep_r.append(0.0)

            loss_critic.append(0.0)
            loss_actor.append(0.0)

            """
            Perform all DDPG updates at the end of the trajectory
            Train on a randomly sampled batch as often there were steps in this episode
            """
            for _ in tqdm(range(ep_l)):
                # lc, la = agent.train()
                lc = agent.train()
                loss_critic[-1] += lc
                # lc = agent.train()
                # loss_actor[-1] += 0.0#la
            loss_critic[-1] /= ep_l
            # loss_actor[-1] /= ep_l

            o, r, d, ep_l = env.reset(), 0, False, 0

            """
            Plotting
            """
            # W = 15
            # if len(ep_r) < W:
            #     W = len(ep_r)
            # sma_ep_r = np.convolve(ep_r, np.ones((W,))/W, mode='full')
            ret_line.set_data(range(len(ep_r)),ep_r)

            lc_line.set_data(range(len(loss_critic)), loss_critic)
            # la_line.set_data(range(len(loss_actor)), loss_actor)

            ax1.set_xlim((0, len(ep_r)))
            ax1.set_ylim((min(ep_r), max(ep_r)))
            ax2.set_xlim((0, len(loss_critic)))
            ax2.set_ylim((min(loss_critic), max(loss_critic)))
            # ax3.set_xlim((0, len(loss_actor)))
            # ax3.set_ylim((min(loss_actor), max(loss_actor)))

            plt.pause(1e-9)
            ep_l = 0

    # Simple moving average over 5 episodes (smoothing) and plot
    sma_rewards = np.convolve(ep_r, np.ones((5,))/5, mode='valid')
    # Plot learning curve

    plt.figure()
    plt.style.use('seaborn')
    plt.plot(sma_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == '__main__':
    train_lunar_lander()




