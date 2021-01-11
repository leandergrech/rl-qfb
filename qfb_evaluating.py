import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import TD3, HER, PPO2
from stable_baselines.her import HERGoalEnvWrapper

from qfb_env import QFBEnv, QFBGoalEnv

if __name__ == '__main__':
    model_name = 'PPO_121220_1754'

    model = PPO2.load(os.path.join('models', model_name))

    # env = HERGoalEnvWrapper(QFBGoalEnv())
    env = QFBEnv()

    n_obs = env.obs_dimension
    n_act = env.act_dimension

    fig, (ax1, ax2) = plt.subplots(2)
    o_x = range(n_obs)
    a_x = range(n_act)

    o_bars = ax1.bar(o_x, np.zeros(n_obs))
    ax1.axhline(0.0, color='k', ls='dashed')
    a_bars = ax2.bar(a_x, np.zeros(n_act))

    for ax, title in zip(fig.get_axes(), ('State', 'Action')):
        ax.set_ylim((-1, 1))
        ax.set_title(title)

    plt.ion()

    n_episodes = 10
    max_steps = 20
    for ep in range(n_episodes):
        o = env.reset()

        o_bars.remove()
        o_bars = ax1.bar(o_x, o, color='b')

        plt.draw()
        plt.pause(0.5)
        for step in range(max_steps):
            fig.suptitle(f'Ep #{ep} - Step #{step}')
            a = model.predict(o)[0]
            o = env.step(a, noise_std=0)[0]

            o_bars.remove()
            a_bars.remove()
            o_bars = ax1.bar(o_x, o, color='b')
            a_bars = ax2.bar(a_x, a, color='r')

            for ax, dat in zip(fig.get_axes(), (o, a)):
                ymin_old, ymax_old = ax.get_ylim()
                ax.set_ylim((np.min(np.concatenate([[ymin_old], dat])), np.max(np.concatenate([[ymax_old], dat]))))

            plt.draw()
            plt.pause(0.5)

        plt.pause(2)


