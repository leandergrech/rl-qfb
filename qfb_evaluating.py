import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from stable_baselines import TD3

from envs.qfb_env import QFBEnv


def plot_individual(model_name):

    model = TD3.load(os.path.join('models', model_name))

    # env = HERGoalEnvWrapper(QFBGoalEnv())
    env = QFBEnv(noise_std=0.0)
    opt_env = QFBEnv(noise_std=0.0)

    n_obs = env.obs_dimension
    n_act = env.act_dimension

    f, axes = plt.subplots(3, 2)
    # fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, num=1)
    fig, ((ax1, ax4), (ax2, ax5)) = plt.subplots(2, 2, num=1)
    fig2, axr = plt.subplots(num=2)
    o_x = range(n_obs)
    a_x = range(n_act)

    # ax1
    o_bars = ax1.bar(o_x, np.zeros(n_obs))
    ax1.axhline(0.0, color='k', ls='dashed')
    ax1.axhline(env.Q_goal, color='g', ls='dashed')
    ax1.axhline(-env.Q_goal, color='g', ls='dashed')
    ax1.set_title('State')
    ax1.set_ylim((-1, 1))
    # ax2
    a_bars = ax2.bar(a_x, np.zeros(n_act))
    ax2.set_title('Action')
    ax2.set_ylim((-1, 1))
    # ax3
    rew_line, = axr.plot([], [], label='Agent')
    axr.set_xlabel('Steps')
    axr.set_ylabel('Reward')
    # ax4
    opt_o_bars = ax4.bar(o_x, np.zeros(n_obs))
    ax4.axhline(0.0, color='k', ls='dashed')
    ax4.axhline(env.Q_goal, color='g', ls='dashed')
    ax4.axhline(-env.Q_goal, color='g', ls='dashed')
    ax4.set_title('Opt State')
    ax4.set_ylim((-1, 1))
    # ax5
    opt_a_bars = ax5.bar(a_x, np.zeros(n_act))
    ax5.set_title('Opt Action')
    ax5.set_ylim((-1, 1))
    # ax6
    opt_rew_line, = axr.plot([], [], label='Optimal')
    axr.axhline(env.objective([env.Q_goal]*2), color='g', ls='dashed', label='Reward threshold')
    axr.set_title('Opt Reward')
    axr.legend(loc='lower right')

    def update_bars(o, a, opo, opa):
        nonlocal o_bars, a_bars, opt_o_bars, opt_a_bars, o_x, a_x
        for bar in (o_bars, a_bars, opt_o_bars, opt_a_bars):
            bar.remove()

        o_bars = ax1.bar(o_x, o, color='b')
        a_bars = ax2.bar(a_x, a, color='r')
        opt_o_bars = ax4.bar(o_x, opo, color='b')
        opt_a_bars = ax5.bar(a_x, opa, color='r')


    plt.ion()

    n_episodes = 10
    max_steps = 50
    for ep in range(n_episodes):
        o = env.reset()
        opt_o = o.copy()
        opt_env.reset(opt_o)

        o_bars.remove()
        a_bars.remove()
        o_bars = ax1.bar(o_x, o, color='b')
        a_bars = ax2.bar(a_x, np.zeros(n_act))

        opt_o_bars.remove()
        opt_a_bars.remove()
        opt_o_bars = ax4.bar(o_x, opt_o, color='b')
        opt_a_bars = ax5.bar(a_x, np.zeros(n_act))

        plt.draw()
        plt.pause(2)

        rewards = []
        opt_rewards = []
        for step in range(max_steps):
            # Put some obs noise to test agent
            # FInd limiting noise
            a = model.predict(o)[0] * 0.1
            o, r, d, _ = env.step(a, noise_std=0.0)
            rewards.append(r)

            opt_a = opt_env.get_optimal_action(opt_o) * 0.1
            opt_o, opt_r, *_ = opt_env.step(opt_a)
            opt_rewards.append(opt_r)

            fig.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
            fig2.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
            update_bars(o, a, opt_o, opt_a)

            rew_line.set_data(range(step + 1), rewards)
            opt_rew_line.set_data(range(step + 1), opt_rewards)
            axr.set_ylim((min(np.concatenate([rewards, opt_rewards])), 0))
            axr.set_xlim((0, step+1))

            if plt.fignum_exists(1) and plt.fignum_exists(2):
                plt.draw()
                plt.pause(0.1)
            else:
                exit()

            if d:
                break

def plot_individual_mask_action():
    model_name = 'TD3_QFB_011321_0058_1000000_steps.zip'
    model = TD3.load(os.path.join('models', model_name))

    # env = HERGoalEnvWrapper(QFBGoalEnv())
    env = QFBEnv(noise_std=0.0)
    opt_env = QFBEnv(noise_std=0.0)

    n_obs = env.obs_dimension
    n_act = env.act_dimension

    action_mask = np.ones(n_act)
    action_mask[5] = 0
    action_mask[10] = 0
    action_mask[11] = 0

    f, axes = plt.subplots(3, 2)
    # fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, num=1)
    fig, ((ax1, ax4), (ax2, ax5)) = plt.subplots(2, 2, num=1)
    fig2, axr = plt.subplots(num=2)
    o_x = range(n_obs)
    a_x = range(n_act)

    # ax1
    o_bars = ax1.bar(o_x, np.zeros(n_obs))
    ax1.axhline(0.0, color='k', ls='dashed')
    ax1.axhline(env.Q_goal, color='g', ls='dashed')
    ax1.axhline(-env.Q_goal, color='g', ls='dashed')
    ax1.set_title('State')
    ax1.set_ylim((-1, 1))
    # ax2
    a_bars = ax2.bar(a_x, np.zeros(n_act))
    ax2.set_title('Action')
    ax2.set_ylim((-1, 1))
    # ax3
    rew_line, = axr.plot([], [], label='Prediction')
    axr.set_title('Reward')
    # ax4
    opt_o_bars = ax4.bar(o_x, np.zeros(n_obs))
    ax4.axhline(0.0, color='k', ls='dashed')
    ax4.axhline(env.Q_goal, color='g', ls='dashed')
    ax4.axhline(-env.Q_goal, color='g', ls='dashed')
    ax4.set_title('Opt State')
    ax4.set_ylim((-1, 1))
    # ax5
    opt_a_bars = ax5.bar(a_x, np.zeros(n_act))
    ax5.set_title('Opt Action')
    ax5.set_ylim((-1, 1))
    # ax6
    opt_rew_line, = axr.plot([], [], label='Optimal')
    axr.set_title('Opt Reward')
    axr.legend(loc='lower right')

    def update_bars(o, a, opo, opa):
        nonlocal o_bars, a_bars, opt_o_bars, opt_a_bars, o_x, a_x
        for bar in (o_bars, a_bars, opt_o_bars, opt_a_bars):
            bar.remove()

        o_bars = ax1.bar(o_x, o, color='b')
        a_bars = ax2.bar(a_x, a, color='r')
        opt_o_bars = ax4.bar(o_x, opo, color='b')
        opt_a_bars = ax5.bar(a_x, opa, color='r')


    plt.ion()

    n_episodes = 10
    max_steps = 50
    for ep in range(n_episodes):
        o = env.reset()
        opt_o = o.copy()
        opt_env.reset(opt_o)

        o_bars.remove()
        a_bars.remove()
        o_bars = ax1.bar(o_x, o, color='b')
        a_bars = ax2.bar(a_x, np.zeros(n_act))

        opt_o_bars.remove()
        opt_a_bars.remove()
        opt_o_bars = ax4.bar(o_x, opt_o, color='b')
        opt_a_bars = ax5.bar(a_x, np.zeros(n_act))

        plt.draw()
        plt.pause(0.5)

        rewards = []
        opt_rewards = []
        for step in range(max_steps):
            a = model.predict(o)[0]
            a = np.multiply(action_mask, a)
            o, r, d, _ = env.step(a, noise_std=0.0)
            rewards.append(r)

            opt_a = opt_env.get_optimal_action(opt_o)
            opt_a = np.multiply(action_mask, opt_a)
            opt_o, opt_r, *_ = opt_env.step(opt_a)
            opt_rewards.append(opt_r)

            fig.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
            update_bars(o, a, opt_o, opt_a)

            rew_line.set_data(range(step + 1), rewards)
            opt_rew_line.set_data(range(step + 1), opt_rewards)
            axr.set_ylim((min(np.concatenate([rewards, opt_rewards])), max(np.concatenate([rewards, opt_rewards]))))
            axr.set_xlim((0, step+1))

            if plt.fignum_exists(1) and plt.fignum_exists(2):
                plt.draw()
                plt.pause(0.01)
            else:
                exit()

            if d:
                break


def eval_set():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_prefix = 'TD3_QFB_011321_0058'
    model_names = []
    models = []
    for item in os.listdir('models'):
        if model_prefix in item:
            model_names.append(item)

    model_names = model_names
    env = QFBEnv(noise_std=0.0)
    print('Loading models')
    for i, name in enumerate(tqdm(sorted(model_names))):
        if i % 1 == 0:
            models.append(TD3.load(os.path.join('models', name), env))
    # models = tqdm([TD3.load(os.path.join('models', name), env) for name in sorted(model_names)])

    nb_eval_eps = 100
    max_steps = 100

    ep_returns = np.zeros((len(models), nb_eval_eps))
    print('Evaluating Models')
    for i, model in enumerate(tqdm(models)):
        for ep in range(nb_eval_eps):
            o = env.reset()
            for step in range(max_steps):
                a = model.predict(o)[0]
                o, r, d, _ = env.step(a)
                ep_returns[i, ep] += r
                if d:
                    break
    ep_returns = np.mean(ep_returns, axis=1)
    fig, ax = plt.subplots()
    ax.plot(ep_returns)
    ax.set_ylabel('Return')
    ax.set_xlabel('Model')
    plt.show()




if __name__ == '__main__':
    # eval_set()
    # plot_individual_mask_action()
    # plot_individual('SAC_QFB_011321_1618_45000_steps.zip')
    plot_individual('TD3_QFB_011321_0058_1000000_steps.zip')