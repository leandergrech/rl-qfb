import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import TD3

from qfb_env import QFBEnv

if __name__ == '__main__':
    model_name = 'TD_121120_2124'

    model = TD3.load(os.path.join('models', model_name))

    env = QFBEnv()

    n_obs = env.obs_dimension
    n_act = env.act_dimension

    fig, (ax1, ax2) = plt.subplots(2)
    o_line, = ax1.plot(np.zeros(n_obs))
    a_line, = ax2.plot(np.zeros(n_act))

    for ax, title in zip(fig.get_axes(), ('State', 'Action')):
        ax.set_ylim((-1, 1))
        ax.set_title(title)

    plt.ion()

    n_episodes = 10
    max_steps = 10
    for ep in range(n_episodes):
        o = env.reset()
        o_line.set_ydata(o)
        plt.draw()
        plt.pause(0.5)
        for step in range(max_steps):
            a = model.predict(o)[0]
            o = env.step(a)[0]

            o_line.set_ydata(o)
            a_line.set_ydata(a)

            for ax, dat in zip(fig.get_axes(), (o, a)):
                ymin_old, ymax_old = ax.get_ylim()
                ax.set_ylim((np.min(np.concatenate([[ymin_old], dat])), np.max(np.concatenate([[ymax_old], dat]))))

            plt.draw()
            plt.pause(0.5)


