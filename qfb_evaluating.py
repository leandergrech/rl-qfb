import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm import tqdm as pbar
import tensorflow as tf
from stable_baselines import TD3, SAC

from envs.qfb_env import QFBEnv
from envs.qfb_env_carnival import QFBEnvCarnival


def plot_individual(model, env, opt_env):
    n_obs = env.obs_dimension
    n_act = env.act_dimension

    fig = plt.figure(figsize=(10, 5), num=1)

    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    axr = fig.add_subplot(gs[:, 2:])

    o_x = range(n_obs)
    a_x = range(n_act)

    # ax1
    o_bars = ax1.bar(o_x, np.zeros(n_obs))
    ax1.axhline(0.0, color='k', ls='dashed')
    ax1.axhline(env.Q_goal, color='g', ls='dashed')
    ax1.axhline(-env.Q_goal, color='g', ls='dashed')
    ax1.set_title('State')
    ax1.set_ylim((-0.2, 0.2))
    # ax2
    a_bars = ax2.bar(a_x, np.zeros(n_act))
    ax2.set_title('Action')
    ax2.set_ylim((-0.1, 0.1))
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
    ax4.set_ylim((-0.2, 0.2))
    # ax5
    opt_a_bars = ax5.bar(a_x, np.zeros(n_act))
    ax5.set_title('Opt Action')
    ax5.set_ylim((-0.1, 0.1))
    # ax6
    opt_rew_line, = axr.plot([], [], label='Optimal')
    axr.axhline(env.objective([env.Q_goal]*2), color='g', ls='dashed', label='Reward threshold')
    axr.set_title('Opt Reward')
    axr.legend(loc='lower right')

    fig.tight_layout()

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
            # Put some obs noise to test agent
            # Find limiting noise
            a = model.predict(o)[0] * 0.1
            o, r, d, _ = env.step(a, noise_std=0.0)
            rewards.append(r)

            opt_a = opt_env.get_optimal_action(opt_o) * 0.1
            opt_o, opt_r, *_ = opt_env.step(opt_a)
            opt_rewards.append(opt_r)

            fig.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
            # fig2.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
            update_bars(o, a, opt_o, opt_a)

            rew_line.set_data(range(step + 1), rewards)
            opt_rew_line.set_data(range(step + 1), opt_rewards)
            axr.set_ylim((min(np.concatenate([rewards, opt_rewards])), 0))
            axr.set_xlim((0, step+1))

            if step < max_steps/3:
                ax1.set_ylim((-0.7, 0.7))
                ax4.set_ylim((-0.7, 0.7))
            elif step < 2*max_steps/3:
                ax1.set_ylim((-0.2, 0.2))
                ax4.set_ylim((-0.2, 0.2))
            else:
                ax1.set_ylim((-0.08, 0.08))
                ax4.set_ylim((-0.08, 0.08))

            if plt.fignum_exists(1):
                plt.draw()
                plt.pause(0.1)
            else:
                exit()

            if d:
                break

def plot_individual_mask_action(model, env, opt_env):
    n_obs = env.obs_dimension
    n_act = env.act_dimension

    action_mask = np.ones(n_act)
    # action_mask[5] = 0
    action_mask[10] = 0
    # action_mask[11] = 0

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

def eval_set(model_prefix, env):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_names = []
    models = []
    for item in os.listdir('models'):
        if model_prefix in item:
            model_names.append(item)

    model_names = model_names
    print('Loading models')
    for i, name in enumerate(tqdm(sorted(model_names))):
        chkpt = int(name.split('_')[-2])
        if chkpt % int(1e3) == 0:
            models.append(SAC.load(os.path.join('models', name), env))
    # models = tqdm([TD3.load(os.path.join('models', name), env) for name in sorted(model_names)])

    nb_eval_eps = 10
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

def plot_rewards_statistics1(models, envs, opt_envs, model_names, colors, noise):
    assert len(models) == len(envs) == len(opt_envs), 'You must provide two environments, one for agent and the other for the optimal, for every model'
    n_models = len(models)
    n_episodes = 20
    n_steps = 50

    rewards = np.zeros(shape=(n_models, n_episodes, n_steps))
    opt_rewards = np.zeros(shape=(n_models, n_episodes, n_steps))
    for ep in pbar(range(n_episodes)):
        o = np.zeros(shape=(n_models, opt_envs[0].obs_dimension))
        opt_o = np.zeros(shape=(n_models, opt_envs[0].obs_dimension))
        for i, env in enumerate(envs):
            o[i] = env.reset()
            opt_o[i] = o[i].copy()
            opt_envs[i].reset(opt_o[i])

        for step in range(n_steps):
            for i, (model, env, opt_env) in enumerate(zip(models, envs, opt_envs)):
                a = model.predict(o[i])[0] * 0.1
                o[i], rewards[i, ep, step], d, _ = env.step(a, noise_std=0.0)

                opt_a = opt_env.get_optimal_action(opt_o[i]) * 0.1
                opt_o[i], opt_rewards[i, ep, step], *_ = opt_env.step(opt_a)

    fig, ax = plt.subplots()

    for rr, orr, c in zip(rewards, opt_rewards, colors):
        for r, o_r in zip(rr, orr):
            ax.plot(r, lw=1, alpha=0.8, color=c)
            ax.plot(o_r, lw=1, alpha=0.8, color=colors[-1])

    rewards_std = np.std(rewards, axis=1)
    rewards_mean = np.mean(rewards, axis=1)
    opt_rewards_std = np.mean(np.std(opt_rewards, axis=1), axis=0)
    opt_rewards_mean = np.mean(np.mean(opt_rewards, axis=1), axis=0)

    x_range = range(n_steps)
    for reward_mean, reward_std, model_name, c in zip(rewards_mean, rewards_std, model_names, colors):
        ax.plot(x_range, reward_mean, c, label=f'{model_name} rewards')
        ax.fill_between(x_range, reward_mean - reward_std, reward_mean + reward_std, color=c, alpha=0.4)

    ax.plot(x_range, opt_rewards_mean, 'b', label='Optimal rewards')

    ax.fill_between(x_range, opt_rewards_mean - opt_rewards_std, opt_rewards_mean + opt_rewards_std, color=colors[-1], alpha=0.4)

    # ax.set_xlim((0, 10))

    ax.legend(loc='lower right')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')

    save_path = os.path.join('results', f'models-vs-optimal_noise-{noise}_neps-{n_episodes}_nsteps-{n_steps}.pdf')
    fig.savefig(save_path)
    print(f'Saved statistics to: {save_path}')

def plot_rewards_statistics2(models, envs, opt_envs, model_names, colors, noise):
    assert len(models) == len(envs) == len(opt_envs), 'You must provide two environments, one for agent and the other for the optimal, for every model'
    n_models = len(models)
    n_episodes = 3

    n_steps = 50

    rewards = np.zeros(shape=(n_models, n_episodes, n_steps))
    opt_rewards = np.zeros(shape=(n_models, n_episodes, n_steps))
    for ep in pbar(range(n_episodes)):
        o = np.zeros(shape=(n_models, opt_envs[0].obs_dimension))
        opt_o = np.zeros(shape=(n_models, opt_envs[0].obs_dimension))
        for i, env in enumerate(envs):
            o[i] = env.reset()
            opt_o[i] = o[i].copy()
            opt_envs[i].reset(opt_o[i])

        for step in range(n_steps):
            for i, (model, env, opt_env) in enumerate(zip(models, envs, opt_envs)):
                a = model.predict(o[i])[0] * 0.1
                o[i], rewards[i, ep, step], d, _ = env.step(a, noise_std=0.0)

                opt_a = opt_env.get_optimal_action(opt_o[i]) * 0.1
                opt_o[i], opt_rewards[i, ep, step], *_ = opt_env.step(opt_a)

    fig, ax = plt.subplots()

    # for rr, orr, c in zip(rewards, opt_rewards, colors):
    #     for r, o_r in zip(rr, orr):
    #         ax.plot(r, lw=0.5, ls='dashed', alpha=0.8, color=c)
    #         ax.plot(o_r, lw=0.5, ls='dashed', alpha=0.8, color=colors[-1])

    rewards_min_manifold = np.array([[np.min(rrr) for rrr in rr.T] for rr in rewards])
    rewards_max_manifold = np.array([[np.max(rrr) for rrr in rr.T] for rr in rewards])

    opt_rewards_min_manifold = np.mean(np.array([[np.min(rrr) for rrr in rr.T] for rr in opt_rewards]), axis=0)
    opt_rewards_max_manifold = np.mean(np.array([[np.max(rrr) for rrr in rr.T] for rr in opt_rewards]), axis=0)

    rewards_med = np.median(rewards, axis=1)
    opt_rewards_med = np.mean(np.median(opt_rewards, axis=1), axis=0)

    x_range = range(n_steps)
    for reward_med, reward_min_man, reward_max_man, model_name, c in zip(rewards_med, rewards_min_manifold, rewards_max_manifold, model_names, colors):
        ax.plot(x_range, reward_med, c, label=f'{os.path.splitext(model_name)[0]} rewards')
        ax.fill_between(x_range, reward_min_man, reward_max_man, color=c, alpha=0.4)

    ax.plot(x_range, opt_rewards_med, 'b', label='Optimal rewards')
    ax.fill_between(x_range, opt_rewards_min_manifold, opt_rewards_max_manifold, color=colors[-1], alpha=0.4)

    # ax.set_xlim((0, 10))

    ax.legend(loc='lower right')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')

    save_path = os.path.join('results', f'models-vs-optimal_noise-{noise}_neps-{n_episodes}_nsteps-{n_steps}_median.pdf')
    fig.savefig(save_path)
    print(f'Saved statistics to: {save_path}')

def save_agent_vs_optimal_gif(model, env, opt_env, model_name, title_name):
    n_steps = 40
    n_eps = 3

    model_name = os.path.splitext(model_name)[0]
    save_path = os.path.join('results', model_name + '.gif')

    n_obs = env.obs_dimension
    n_act = env.act_dimension
    
    o_x = range(n_obs)
    a_x = range(n_act)

    init_state = np.zeros(n_obs)
    init_action = np.zeros(n_act)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title_name)
    gs = fig.add_gridspec(3, 2)

    ax_state = fig.add_subplot(gs[0, 0])
    ax_action = fig.add_subplot(gs[1, 0])
    ax_state_opt = fig.add_subplot(gs[0, 1])
    ax_action_opt = fig.add_subplot(gs[1, 1])
    ax_rewards = fig.add_subplot(gs[2, :])

    plt.subplots_adjust(left=0.09, bottom=0.08, right=.95, top=.87, wspace=0.2, hspace=0.2)

    for i, ax in enumerate(fig.axes):
        if i != len(fig.axes) - 1:
            ax.grid(axis='y')

    # ax_state
    o_bars = ax_state.bar(o_x, init_state)
    ax_state.axhline(0.0, color='k', ls='dashed')
    ax_state.axhline(env.Q_goal, color='g', ls='dashed', label='Goal')
    ax_state.axhline(-env.Q_goal, color='g', ls='dashed')
    ax_state.set_title('State - Agent')
    ax_state.set_ylim((-1, 1))

    # ax_action
    a_bars = ax_action.bar(a_x, init_action)
    ax_action.set_title('Action - Agent')
    ax_action.set_ylim((-0.1, 0.1))

    # ax_state_opt
    o_bars_opt = ax_state_opt.bar(o_x, init_state)
    ax_state_opt.axhline(0.0, color='k', ls='dashed')
    ax_state_opt.axhline(env.Q_goal, color='g', ls='dashed', label='Goal')
    ax_state_opt.axhline(-env.Q_goal, color='g', ls='dashed')
    ax_state_opt.set_title('State - Optimal')
    ax_state_opt.set_ylim((-1, 1))

    # ax_action_opt
    a_bars_opt = ax_action_opt.bar(a_x, init_action)
    ax_action_opt.set_title('Action - Optimal')
    ax_action_opt.set_ylim((-0.1, 0.1))

    # ax_rewards
    rew_line, = ax_rewards.plot([], [], label='Rewards - Agent')
    rew_line_opt, = ax_rewards.plot([], [], label='Rewards - Optimal')
    ax_rewards.set_xlabel('Steps')
    ax_rewards.set_ylabel('|Reward|')
    ax_rewards.set_yscale('log')
    ax_rewards.grid(which='both')

    for i, ax in enumerate(fig.axes):
        ax.legend(loc='upper right')

    o = env.reset()
    opt_o = o.copy()
    opt_env.reset(opt_o)
    a, opt_a = init_action, init_action

    o_bars.remove()
    a_bars.remove()
    o_bars_opt.remove()
    a_bars_opt.remove()

    o_bars = ax_state.bar(o_x, o, color='b')
    a_bars = ax_action.bar(a_x, a, color='r')
    o_bars_opt = ax_state_opt.bar(o_x, opt_o, color='b')
    a_bars_opt = ax_action_opt.bar(a_x, opt_a, color='r')

    def update_bars(o, a, opo, opa):
        nonlocal o_bars, a_bars, o_bars_opt, a_bars_opt, o_x, a_x
        nonlocal ax_state, ax_action, ax_state_opt, ax_action_opt

        # for bar in (o_bars, a_bars, o_bars_opt, a_bars_opt):
        #     bar.remove()

        for bar, dat in zip((o_bars, a_bars, o_bars_opt, a_bars_opt), (o, a, opo, opa)):
            for b, d in zip(bar, dat):
                b.set_height(d)

    rewards = []
    rewards_opt = []
    rew_x_delta = 1/n_steps
    rew_x = []
    rew_x_cntr = 0.0
    def animate(i):
        nonlocal o_bars, a_bars, o_bars_opt, a_bars_opt
        nonlocal ax_state, ax_state_opt
        nonlocal a, o, opt_a, opt_o
        nonlocal rewards, rewards_opt, rew_x_cntr

        ep_idx = int(i / n_steps)
        step_idx = i % n_steps

        d = False

        rew_x_cntr += rew_x_delta
        rew_x.append(rew_x_cntr)

        if i % n_steps == 0:
            o = env.reset()
            opt_o = o.copy()
            opt_env.reset(opt_o)
            a, opt_a = init_action, init_action
            update_bars(o, a, opt_o, opt_a)

            rewards.append(-env.objective(o))
            rewards_opt.append(-env.objective(opt_o))

        else:
            a = model.predict(o)[0] * 0.1
            o, r, d, _ = env.step(a)

            opt_a = opt_env.get_optimal_action(opt_o) * 0.1
            opt_o, opt_r, *_ = opt_env.step(opt_a)

            rewards.append(-r)
            rewards_opt.append(-opt_r)

        rew_line.set_data(rew_x, rewards)
        rew_line_opt.set_data(rew_x, rewards_opt)
        ax.set_xlim((0, rew_x[-1]))
        ax.set_ylim((min(np.concatenate([rewards, rewards_opt])), max(np.concatenate([rewards, rewards_opt]))))

        ax_state.set_title(f'Ep #{ep_idx} - Step #{step_idx} - Done {d}\nState - Agent')
        ax_state_opt.set_title(f'Ep #{ep_idx} - Step #{step_idx} - Done {d}\nState - Optimal')
        update_bars(o, a, opt_o, opt_a)

        # return (o_bars, a_bars, o_bars_opt, a_bars_opt,)

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=n_steps * n_eps,
                            interval=100, blit=False, repeat=False)

    # plt.show()
    anim.save(save_path)
    print('Saved to: ', save_path)


def main():

    # for noise in [0.5, 0.25, 0.1, 0.05, 0.01]:
    noise = 0.1

    model_names = ['SAC_QFB_030321_2058_100000_steps.zip']
        # 'SAC_QFB_011321_1618_90000_steps.zip',]
                   # 'TD3_QFB_011321_0058_90000_steps.zip']
    title_names = [f'SAC QFB steps=90000 action-noise={noise*100.0}%']
    model_types = [SAC, TD3]
    models = []
    for i, (model_name, model_type) in enumerate(zip(model_names, model_types)):
        models.append(model_type.load(os.path.join('models', model_name)))

    env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
                      calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'))
    envs = [QFBEnv(noise_std=noise, **env_kwargs) for _ in range(len(model_names))]
    opt_envs = [QFBEnv(noise_std=noise, **env_kwargs) for _ in range(len(model_names))]

    plot_individual(models[0], envs[0], opt_envs[0])
    # plot_rewards_statistics2(models, envs, opt_envs, model_names, ['r', 'tab:orange', 'b'], noise)
    # save_agent_vs_optimal_gif(models[0], envs[0], opt_envs[0], model_names[0], title_names[0])
    # model_prefix = 'SAC_QFB_011321_1618'
    # eval_set(model_prefix, env)


if __name__ == '__main__':
    main()