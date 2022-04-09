import os
import random
from copy import copy
from pandas import DataFrame as df
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if int(tf.__version__.split('.')[0]) == 1:  # stable-baselines only works with TF1
	from stable_baselines import TD3, SAC, PPO2

# from qfb_env.qfb_env import QFBEnv
from qfb_env.envs import QFBNLEnv

# from qfb_env.qfb_env_carnival import QFBEnvCarnival

agent_state_kw = dict(color='b', ls='solid', lw=1.5, marker='x')#, alpha=0.6)
agent_state2_kw = dict(color='b', ls='solid', lw=0.5, marker='^', markersize=2)#, alpha=0.6)
agent_action_kw = dict(color='r', ls='solid', lw=0.5)#, alpha=0.8)

pi_state_kw = dict(color='c', ls='solid', lw=1.5, marker='x')
pi_state2_kw = dict(color='c', ls='solid', marker='^', markersize=2, lw=0.5)
pi_action_kw = dict(color='m', ls='dotted', lw=0.8)
threshold_lw = 1.5
threshold_style = 'dotted'

fig_size = (10, 7)
legend_font_size = 16
legend_font_weight = 'ultralight'
legend_ncol = 3
axis_label_font_size = 18
axis_label_font_weight = 'ultralight'
axis_label_font_style = 'italic'
title_font_size = 20
title_font_weight = 'ultralight'
subtitle_font_size = 18
subtitle_font_weight = 'ultralight'
tests_subplot_adjust_kwargs = dict(bottom=0.23, wspace=0.21, hspace=0.26, top=0.90, left=0.095, right=0.98)

# actfail_fig_size = (15, 6.5)
actfail_fig_size = (10, 7)
actfail_legend_ncol = 3
actfail_color = 'xkcd:brick'
pifail_color = 'xkcd:plum'
actfail_subtitle_font_size = 12
actfail_subtitle_font_weight = 'ultralight'
agent_actfail_kw = dict(color=actfail_color, ls='--', lw=1.5, alpha=1)
pi_actfail_kw = dict(color=pifail_color, ls='dashdot', lw=1.5, alpha=1)
# actfail_tests_subplot_adjust_kwargs = dict(bottom=0.27, wspace=0.21, hspace=0.36, top=0.85, left=0.06, right=0.99)
actfail_tests_subplot_adjust_kwargs = dict(bottom=0.285, wspace=0.21, hspace=0.36, top=0.85, left=0.08, right=0.99)

episode_marker_size = 25
episode_marker_lw = 1.2
episode_start_color = 'xkcd:tree green'
episode_end_color = 'xkcd:deep red'


def save_agent_vs_optimal_gif(model, env, opt_env, model_name, title_name):
	def get_action_ylims(action):
		discrete_val = 0.2

		mina, maxa = -np.abs(min(action)), np.abs(max(action))

		ylim_min = np.floor(mina / discrete_val) * discrete_val
		ylim_max = np.ceil(maxa / discrete_val) * discrete_val

		return (ylim_min, ylim_max)

	n_steps = 40
	n_eps = 3

	model_name = os.path.splitext(model_name)[0]
	save_path = os.path.join('results', 'gifs', model_name + '.gif')

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
	ax_action.set_ylim(get_action_ylims(init_action))

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
	ax_action_opt.set_ylim(get_action_ylims(init_action))

	# ax_rewards
	rew_line, = ax_rewards.plot([], [], label='Rewards - Agent')
	rew_line_opt, = ax_rewards.plot([], [], label='Rewards - Optimal')
	ax_rewards.axhline(abs(env.reward_thresh), color='g', ls='dashed', label='Reward threshold')
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
	rew_x_delta = 1 / n_steps
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
			a = model.predict(o)[0]
			o, r, d, _ = env.step(a)

			opt_a = opt_env.pi_controller_action()
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
		ax_action.set_ylim(get_action_ylims(a))
		ax_action_opt.set_ylim(get_action_ylims(a))

	# return (o_bars, a_bars, o_bars_opt, a_bars_opt,)

	anim = animation.FuncAnimation(fig=fig, func=animate, frames=n_steps * n_eps,
	                               interval=100, blit=False, repeat=False)

	# plt.show()
	anim.save(save_path)
	print('Saved to: ', save_path)


# def save_episode_trace(model, env, opt_env, model_name, title_name):
# 	o = env.reset()
# 	opt_env.reset(o.copy())
#
# 	n_act = env.act_dimension
#
# 	actions_model = [np.zeros(n_act)]
# 	actions_pi = [np.zeros(n_act)]
# 	states_model = [o.copy()]
# 	states_pi = [o.copy()]
# 	rewards_model = []
# 	rewards_pi = []
#
# 	# Play an episode and store transitions
# 	for step in range(env.EPISODE_LENGTH_LIMIT):
# 		a_model = model.predict(o.copy(), deterministic=True)[0]
# 		a_pi = opt_env.pi_controller_action()
#
# 		o, r, d_model, _ = env.step(a_model)
# 		o_pi, r_pi, d_pi, _ = opt_env.step(a_pi)
#
# 		rewards_model.append(r)
# 		rewards_pi.append(r_pi)
#
# 		actions_model.append(a_model)
# 		actions_pi.append(a_pi)
# 		states_model.append(o.copy())
# 		states_pi.append(o_pi.copy())
#
# 		if d_model and d_pi:
# 			break
#
# 	# Plot the shit
# 	fig = plt.figure(figsize=(8, 8))
# 	fig.suptitle(title_name)
# 	gs = fig.add_gridspec(3, 2)
#
# 	ax_state = fig.add_subplot(gs[0, 0])
# 	ax_action = fig.add_subplot(gs[1, 0])
# 	ax_state_opt = fig.add_subplot(gs[0, 1])
# 	ax_action_opt = fig.add_subplot(gs[1, 1])
# 	ax_rewards = fig.add_subplot(gs[2, :])
#
# 	x = np.arange(step + 1)
#
# 	ax_state.set_title('Agent state')
# 	ax_state.plot(np.array(states_model), color='b')
# 	ax_state_opt.set_title('PI state')
# 	ax_state_opt.plot(np.array(states_pi), color='b')
# 	for ax in (ax_state, ax_state_opt):
# 		ax.axhline(env.Q_goal, color='g', ls='dashed', label='State threshold')
# 		ax.axhline(-env.Q_goal, color='g', ls='dashed')
# 	ax_action.set_title('Agent action')
# 	ax_action.plot(np.array(actions_model), color='r')
# 	ax_action_opt.set_title('PI action')
# 	ax_action_opt.plot(np.array(actions_pi), color='r')
# 	ax_action_opt.set_ylim(ax_action.get_ylim())
#
# 	ax_rewards.plot(rewards_model, label='Agent Rewards')
# 	ax_rewards.plot(rewards_pi, label='PI Rewards')
# 	ax_rewards.axhline(env.reward_thresh, color='g', ls='dashed', label='Reward threshold')
# 	# ax_rewards.set_yscale('symlog')
#
# 	for ax in fig.axes:
# 		ax.legend(loc='best')
# 		ax.set_xlabel('Steps')
#
# 	fig.tight_layout()
# 	save_path = os.path.join('results', 'policy_tests', 'no_noise', title_name.split('_')[0],
# 	                         os.path.splitext(model_name)[0] + '.pdf')
# 	# fig.savefig(save_path)
# 	print(f'Saved {model_name} to {save_path}')
# 	plt.show()


# def init_NAF2(env):
#     training_info = dict(polyak=0.999,
#                          batch_size=100,
#                          steps_per_batch=10,
#                          epochs=1,
#                          learning_rate=1e-3,
#                          discount=0.9999)
#     nafnet_info = dict(hidden_sizes=[50, 50],
#                        activation=tf.nn.relu,
#                        kernel_initializer=tf.random_normal_initializer(0, 0.05, seed=123))
#     params = dict(buffer_size=int(5e3),
#                   q_smoothing_sigma=0.02,
#                   q_smoothing_clip=0.05)
#
#     # linearly decaying noise function
#     noise_episode_thresh = 40
#     n_act = 16
#     noise_fn = lambda act, i: act + np.random.randn(n_act) * max(1 - i / noise_episode_thresh, 0)
#     agent = NAF2(env=env,
#                  buffer_size=params['buffer_size'],
#                  train_every=1,
#                  training_info=training_info,
#                  eval_info={},
#                  save_frequency=200,
#                  log_frequency=10,
#                  directory=None,
#                  tb_log=None,
#                  q_smoothing_sigma=params['q_smoothing_sigma'],
#                  q_smoothing_clip=params['q_smoothing_clip'],
#                  nafnet_info=nafnet_info,
#                  noise_fn=noise_fn)
#
#     return agent

# def populate_ax_with_evaluation_episode_trace(ax, action_agent, state_agent, action_pi, state_pi):
# 	x_range = np.arange(action_agent.shape[0])
#
# 	ax2 = ax.twinx()
# 	ax2.axhline(0.0, color='r', lw=0.5)
# 	ax2.plot(x_range, np.array(action_agent), c='r', lw=0.5, alpha=0.6, label='Agent actions')
# 	ax2.plot(x_range, np.array(action_pi), c='m', lw=1.0, ls='dotted', alpha=0.6, label='PI actions')
#
# 	ax.plot(x_range, np.array(state_agent), c='b', alpha=0.7, lw=1.5, label='Agent states')
# 	ax.plot(x_range, np.array(state_pi), c='c', lw=1.5, label='PI states')
#
# 	ax.set_xlabel('Steps')
# 	# ax.set_ylabel('$\Delta Q$', color='b')
# 	# ax2.set_ylabel('$\sigma_Q$', color='r')
# 	ax.tick_params(axis='y', labelcolor='b')
# 	ax2.tick_params(axis='y', labelcolor='r')
#
# 	ax.grid(which='both')
#
# 	ax.set_zorder(ax2.get_zorder() + 1)
# 	ax.patch.set_visible(False)


def populate_ax_with_evaluation_episode_trace_2d_tune(axs, action_agent, state_agent, action_pi, state_pi,
                                                      has_all_labels=True):
	ax1 = axs[0]

	state_agent = state_agent.T
	state_pi = state_pi.T

	ax1.plot(state_agent[0], state_agent[1], **agent_state_kw, label='Agent state')
	ax1.plot(state_pi[0], state_pi[1], **pi_state_kw, label='PI state')

	lim_ver = [func(np.concatenate([state_agent[1], state_pi[1]])) for func in [np.min, np.max]]
	lim_hor = [func(np.concatenate([state_agent[0], state_pi[0]])) for func in [np.min, np.max]]

	DIFF = 0.1  # smallest state zoomed
	BUFF = 0.1  # % of limits difference to use as edge buffers
	ver_diff = abs(np.diff(lim_ver))
	if ver_diff < DIFF:
		lim_ver[0] -= (DIFF - ver_diff) / 2
		lim_ver[1] += (DIFF - ver_diff) / 2
	else:
		lim_ver[0] -= abs(BUFF * ver_diff)
		lim_ver[1] += abs(BUFF * ver_diff)
	hor_diff = abs(np.diff(lim_hor))
	if hor_diff < DIFF:
		lim_hor[0] -= (DIFF - hor_diff) / 2
		lim_hor[1] += (DIFF - hor_diff) / 2
	else:
		lim_hor[0] -= abs(BUFF * hor_diff)
		lim_hor[1] += abs(BUFF * hor_diff)
	ax1.set_ylim(lim_ver)
	ax1.set_xlim(lim_hor)

	ax1.scatter([state_agent[0][0]], [state_agent[1][0]], s=episode_marker_size, marker='o', color='None',
	            linewidth=episode_marker_lw, edgecolor=episode_start_color, zorder=10)
	ax1.scatter([state_agent[0][-1]], [state_agent[1][-1]], s=episode_marker_size, marker='o', color='None',
	            linewidth=episode_marker_lw, edgecolor=episode_end_color, zorder=10)
	# ax1.scatter([state_pi[0][0]], [state_pi[1][0]], s=10, marker='s', color='None', edgecolor='b', zorder=10)
	ax1.scatter([state_pi[0][-1]], [state_pi[1][-1]], s=episode_marker_size, marker='s', color='None',
	            linewidth=episode_marker_lw, edgecolor=episode_end_color, zorder=10)

	if has_all_labels:
		ax1.set_ylabel('$\Delta Q_V$ (norm.)', size=axis_label_font_size, weight=axis_label_font_weight,
		               style=axis_label_font_style)
	ax1.set_xlabel('$\Delta Q_H$ (norm.)', size=axis_label_font_size, weight=axis_label_font_weight,
	               style=axis_label_font_style)
	# ax1.set_xlim((-1.2, 1.2))
	# ax1.set_ylim((-1.2, 1.2))
	# ax.text(-1, 0.1, f'Agent ep_len = {len(action_agent)}')
	# ax.text(0, 0.05, f'PI ep_len = {len(action_pi)}')
	# ax.legend(loc='best')

	ax2 = axs[1]

	# ax2.axhline(0.0, color='r', lw=0.5)
	ax2.plot(np.arange(action_agent.shape[0]), np.array(action_agent), c='r', lw=0.5)#, alpha=0.6)
	ax2.plot(np.arange(action_pi.shape[0]), np.array(action_pi), c='m', lw=1.0, ls='dotted')#, alpha=0.6)

	if has_all_labels:
		ax2.set_ylabel('Action (norm.)', size=axis_label_font_size, weight='light')
	ax2.set_xlabel('Steps', size=axis_label_font_size, weight='light')

	for ax in axs:
		ax.grid(which='major', axis='both')
		ax.minorticks_on()

	ax1.grid(which='minor', axis='both')#, alpha=0.3)
	ax2.grid(which='minor', axis='x')#, alpha=0.3)


def populate_ax_with_evaluation_episode_trace_2d_tune_highlight_failure(axs, action_agent, state_agent, state_agent2,
                                                                        action_pi, state_pi, state_pi2, fail_indices,
                                                                        has_all_labels=True):
	ax1 = axs[0]

	state_agent = state_agent.T
	state_agent2 = state_agent2.T
	state_pi = state_pi.T
	state_pi2 = state_pi2.T

	ax1.plot(state_agent[0], state_agent[1], **agent_state_kw, label='Agent state')
	ax1.plot(state_agent2[0], state_agent2[1], **agent_state2_kw)
	ax1.plot(state_pi[0], state_pi[1], **pi_state_kw, label='PI state')
	ax1.plot(state_pi2[0], state_pi2[1], **pi_state2_kw)

	lim_ver = [func(np.concatenate([state_agent[1], state_pi[1]])) for func in [np.min, np.max]]
	lim_hor = [func(np.concatenate([state_agent[0], state_pi[0]])) for func in [np.min, np.max]]

	DIFF = 0.1  # smallest state zoomed
	BUFF = 0.1  # % of limits difference to use as edge buffers
	ver_diff = abs(np.diff(lim_ver))
	if ver_diff < DIFF:
		lim_ver[0] -= (DIFF - ver_diff) / 2
		lim_ver[1] += (DIFF - ver_diff) / 2
	else:
		lim_ver[0] -= abs(BUFF * ver_diff)
		lim_ver[1] += abs(BUFF * ver_diff)
	hor_diff = abs(np.diff(lim_hor))
	if hor_diff < DIFF:
		lim_hor[0] -= (DIFF - hor_diff) / 2
		lim_hor[1] += (DIFF - hor_diff) / 2
	else:
		lim_hor[0] -= abs(BUFF * hor_diff)
		lim_hor[1] += abs(BUFF * hor_diff)
	ax1.set_ylim(lim_ver)
	ax1.set_xlim(lim_hor)

	# Episode start and finish of agent and pi controller
	ax1.scatter([state_agent[0][0]], [state_agent[1][0]], s=episode_marker_size * 1.4, marker='o', color='None',
	            edgecolor=episode_start_color, zorder=10, linewidth=episode_marker_lw)
	ax1.scatter([state_agent[0][-1]], [state_agent[1][-1]], s=episode_marker_size * 1.4, marker='o', color='None',
	            edgecolor=episode_end_color, zorder=10, linewidth=episode_marker_lw)
	# ax1.scatter([state_pi[0][0]], [state_pi[1][0]], s=10, marker='s', color='None', edgecolor='b', zorder=10)
	ax1.scatter([state_pi[0][-1]], [state_pi[1][-1]], s=episode_marker_size, marker='s', color='None',
	            edgecolor=episode_end_color, zorder=10, linewidth=episode_marker_lw)

	if has_all_labels:
		ax1.set_ylabel('$\Delta Q_V$ (norm.)', size=axis_label_font_size, weight=axis_label_font_weight,
		               style=axis_label_font_style)
	ax1.set_xlabel('$\Delta Q_H$ (norm.)', size=axis_label_font_size, weight=axis_label_font_weight,
	               style=axis_label_font_style)
	# ax1.set_xlim((-1.2, 1.2))
	# ax1.set_ylim((-1.2, 1.2))
	# ax.text(-1, 0.1, f'Agent ep_len = {len(action_agent)}')
	# ax.text(0, 0.05, f'PI ep_len = {len(action_pi)}')
	# ax.legend(loc='best')

	ax2 = axs[1]

	# ax2.axhline(0.0, color='r', lw=0.5)
	good_indices = [item for item in np.arange(action_agent.shape[1]) if item not in fail_indices]
	ax2.plot(np.arange(action_agent.shape[0]), np.array(action_agent[:, good_indices]), **agent_action_kw)
	ax2.plot(np.arange(action_pi.shape[0]), np.array(action_pi[:, good_indices]), **pi_action_kw)
	ax2.plot(np.arange(action_pi.shape[0]), np.array(action_pi[:, fail_indices]), **pi_actfail_kw)
	ax2.plot(np.arange(action_agent.shape[0]), np.array(action_agent[:, fail_indices]), **agent_actfail_kw)

	if has_all_labels:
		ax2.set_ylabel('Action (norm.)', size=axis_label_font_size, weight=axis_label_font_weight,
		               style=axis_label_font_style)
	ax2.set_xlabel('Steps', size=axis_label_font_size, weight=axis_label_font_weight, style=axis_label_font_style)

	for ax in axs:
		ax.grid(which='major', axis='both')
		ax.minorticks_on()

	ax1.grid(which='minor', axis='both', alpha=0.3)
	ax2.grid(which='minor', axis='x', alpha=0.3)


def get_best_models(models_dir, model_type, env, txt_filename_suffix=''):
	NB_EPS = 50
	HOW_MANY = 10
	STEP_LIM = 1000000

	if 'AE-DYNA' in models_dir:
		rl_type = os.path.split(os.path.split(models_dir)[0])[-1].split('_')[0]
	else:
		rl_type = os.path.split(models_dir)[-1].split('_')[0]
	n_act = env.act_dimension
	action_noise_sigma = 0.0
	noise = lambda: np.random.normal(0.0, action_noise_sigma, env.act_dimension)

	model_paths = []
	model_steps = []

	best_ep_lens_mean = [env.EPISODE_LENGTH_LIMIT] * HOW_MANY
	best_ep_lens_std = [0.0] * HOW_MANY
	best_models = [None] * HOW_MANY
	best_model_training_steps = [None] * HOW_MANY

	for file in os.listdir(models_dir):
		# if '.zip' not in file and 'step' not in file:
		if 'step' not in file and 'AE-DYNA' not in rl_type or 'initial' in file:
			continue

		if rl_type in 'NAF2':
			step = int(file.split('_')[-1])
		# if step % 1000 != 0:
		#     continue
		elif rl_type in 'AE-DYNA':
			try:
				step = int(file.split('_')[0][9:])

			except Exception as e:
				print(file, e)
				exit(66)

		else:
			step = int(file.split('_')[-2])

		if step > STEP_LIM:
			continue

		model_steps.append(step)

		if 'AE-DYNA' in rl_type:
			model_paths.append(os.path.join(models_dir, file))
		else:
			model_paths.append(os.path.join(models_dir, file))

		model = model_type.load(model_paths[-1])
		if 'TFL' in rl_type or 'AE-DYNA' in rl_type:
			model.policy_net(env.reset().reshape((1, -1)).astype(np.float32))  # only for SACTFL

		print(f'-> Looking in {file}')
		model_ep_lens = []
		successes = []
		for ep in range(NB_EPS):
			if (ep + 1) % 20 == 0:
				print(f' `-> Eval episode {ep}')
			o = env.reset()

			states_model = o.copy().reshape((1, -1))
			actions_model = np.zeros(shape=(1, n_act))
			rewards_model = []

			d_model = False

			for step in range(env.EPISODE_LENGTH_LIMIT):
				if not d_model:
					if 'TFL' in rl_type or 'AE-DYNA' in rl_type:
						a_model = model.predict(o.copy(), deterministic=True) + noise()  # only for SACTFL
					else:
						a_model = model.predict(o.copy(), deterministic=True)[0] + noise()
					a_model = np.clip(a_model, -1.0, 1.0)
					o, r, d_model, _ = env.step(a_model)
					actions_model = np.vstack([actions_model, a_model])
					states_model = np.vstack([states_model, o.copy()])
					rewards_model.append(r)

				if d_model:
					break
			ep_len = len(rewards_model)
			success = 1 if ep_len < env.EPISODE_LENGTH_LIMIT else 0

			model_ep_lens.append(ep_len)

			successes.append(success)
			if ep == 0 and not success:
				print(f'Failed on episode 0. Stopping evaluation and assuming the worst')
				break

		last_model_average_ep_len = np.mean(model_ep_lens)
		last_model_std_ep_len = np.std(model_ep_lens)
		if last_model_average_ep_len < np.max(best_ep_lens_mean):
			idx = np.argmax(best_ep_lens_mean)
			best_models[idx] = model_paths[-1]
			best_model_training_steps[idx] = model_steps[-1]
			best_ep_lens_mean[idx] = last_model_average_ep_len
			best_ep_lens_std[idx] = last_model_std_ep_len

			print(f' `-> Added to best models: {best_models[idx]} ')

	with open(os.path.join('results', 'best_models', f'info_{rl_type}{txt_filename_suffix}.txt'), 'a') as f:
		f.write(f'-> {models_dir}\n')
		for b, em, es in zip(best_model_training_steps, best_ep_lens_mean, best_ep_lens_std):
			if b is not None:
				f.write(f' `-> Training steps={b}, ep_len={em} $\pm$ {es}\n')
		f.write('\n')

	print(f'Best {len(best_models)} models in {models_dir}: {best_models}')


def save_episode_trace_for_diff_train_steps_action_noise(model_path, steps_list, model_type, env, opt_env, title_name,
                                                         action_noise_sigma=0.0, save_dir=None):
	noise = lambda: np.random.normal(0.0, action_noise_sigma, env.act_dimension)

	n_act = opt_env.act_dimension

	rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'

	'''stable-baselines'''
	models = [model_type.load(os.path.join(model_path,
	                                       f'{os.path.split(model_path)[-1]}_{s}_steps.zip')) for s in steps_list]
	'''SAC-TFL'''
	# models = [model_type.load(os.path.join(model_path,
	#                                        f'{os.path.split(model_path)[-1]}_{s}_steps')) for s in steps_list]
	# need an extra call here to make inside functions be able to use model.forward
	# [m.policy_net(env.reset().reshape((1, -1)).astype(np.float32)) for m in models] # only for SACTFl
	'''NAF2'''
	# models = [model_type.load(os.path.join(model_path,
	#                                        f'step_{f"{s}".zfill(4)}')) for s in steps_list]

	# Plot the shit
	fig, axs = plt.subplots(2, 3, figsize=fig_size)
	fig.subplots_adjust(bottom=0.17, wspace=0.3, hspace=0.255, top=0.9, left=0.07, right=0.99)
	axs = axs.flatten()
	fig.suptitle(title_name, size=title_font_size)

	miny, maxy = np.inf, -np.inf
	prev_init_o = []
	i = 0
	for model in models:
		while True:
			o = env.reset()
			if len(prev_init_o) > 0:
				temp = []
				for _o in prev_init_o:
					if np.mean(np.abs(np.subtract(_o, o))) < 0.2:
						continue
			else:
				break

			prev_init_o.append(o)
			break

		opt_env.reset(o.copy())
		opt_env.pi_controller_warm_up()

		states_model = o.copy().reshape((1, -1))
		states_pi = states_model.copy()
		actions_model = np.zeros(shape=(1, n_act))
		actions_pi = np.zeros_like(actions_model)
		rewards_model = []
		rewards_pi = []

		d_model = False
		d_pi = False

		for step in range(env.EPISODE_LENGTH_LIMIT):
			if not d_model:
				a_model = model.predict(o.copy(), deterministic=True)[0] + noise()
				# a_model = model.predict(o.copy(), deterministic=True) + noise() # only for SACTFL
				a_model = np.clip(a_model, -1.0, 1.0)
				o, r, d_model, _ = env.step(a_model)
				actions_model = np.vstack([actions_model, a_model])
				states_model = np.vstack([states_model, o.copy()])
				rewards_model.append(r)

			if not d_pi:
				a_pi = opt_env.pi_controller_action() + noise()
				a_pi = np.clip(a_pi, -1.0, 1.0)
				o_pi, r_pi, d_pi, _ = opt_env.step(a_pi)
				actions_pi = np.vstack([actions_pi, a_pi])
				states_pi = np.vstack([states_pi, o_pi.copy()])
				rewards_pi.append(r_pi)

			if d_model and d_pi:
				# ax = axs[i]
				axes = [axs[i], axs[i + 3]]

				ax = axes[0]
				if i < 2:
					ax.set_title(f'{rl_type} {steps_list[i]} training steps')
				else:
					ax.set_title(f'{rl_type} {steps_list[i]} training steps (best)')
				ax.add_patch(plt.Circle((0, 0), radius=env.Q_goal,
				                        fill=False, edgecolor='g', ls='dashed',
				                        label='Threshold', lw=1, zorder=10))

				populate_ax_with_evaluation_episode_trace_2d_tune(axes, action_agent=actions_model,
				                                                  state_agent=states_model,
				                                                  action_pi=actions_pi, state_pi=states_pi)
				i += 1
				break

	from matplotlib.lines import Line2D
	from matplotlib.patches import Patch

	_lines = [Line2D([0], [0], **kw) for kw in (agent_state_kw, pi_state_kw, agent_action_kw, pi_action_kw)]
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_start_color, marker='o', linewidth=episode_marker_lw))
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color, marker='o', linewidth=episode_marker_lw))
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color, marker='s', linewidth=episode_marker_lw))
	_lines.append(Patch(facecolor='w', edgecolor='g', ls=threshold_style, lw=threshold_lw))

	fig.legend(_lines, ['Agent states', 'PI states', 'Agent actions', 'PI actions',
	                    'Episode start', ' Agent episode end', 'PI episode end',
	                    'State threshold'], ncol=5, loc='lower center', frameon=False, prop={'size': legend_font_size,
	                                                                                         'weight': legend_font_weight})
	# fig.tight_layout()

	model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
	rl_type = model_name.split('_')[0]
	if action_noise_sigma == 0.0:
		noise_name = 'no-action-noise'
	else:
		noise_name = f'{action_noise_sigma * 100.0:.1f}-action-noise'

	if save_dir is None:
		save_dir = os.path.join('results', 'policy_tests')
	elif not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_name = f'{model_name}_{noise_name}'

	file_cnt = 1
	while True:
		save_path = os.path.join(save_dir, f'{save_name}_{file_cnt}.ps')
		if os.path.exists(save_path):
			file_cnt += 1
			continue
		else:
			break

	fig.savefig(save_path)
	print(f'Saved {model_path} to {save_path}')


'''
STATE PERTURBATION WITH 50 HZ HARMONIC NOISE TEST
'''


def save_episode_trace_only_best_state_perturb(model_path, best_step, name_parser, model_type, env, opt_env, title_name,
                                               action_noise_sigma=0.0, save_dir=None):
	NB_EPS = 3
	INIT_OBS_MIN_DIST = 0.6
	VISUAL_THRESH_MULTIPLIER = 2
	LIM_STEPS = 70

	noise = lambda: np.random.normal(0.0, action_noise_sigma, env.act_dimension)

	n_act = opt_env.act_dimension

	rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'
	model_name = os.path.split(model_path)[-1]

	# if 'AE-DYNA' in rl_type:
	#     model = model_type.load(os.path.join(model_path, 'agent', name_parser(model_path, best_step)))
	#     model.policy_net(env.reset().reshape((1, -1)).astype(np.float32)) # only for SACTFl
	#
	# else:
	#     model = model_type.load(os.path.join(model_path, name_parser(model_name, best_step)))
	#     if 'SAC-TFL' in rl_type:
	#         model.policy_net(env.reset().reshape((1, -1)).astype(np.float32)) # only for SACTFl
	full_model_path = os.path.join(model_path, name_parser(model_path, best_step))
	model = model_type.load(full_model_path)
	if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
		model.policy_net(env.reset().reshape((1, -1)).astype(np.float32))  # only for SAC-TFl

	# '''stable-baselines'''
	# model = model_type.load(os.path.join(model_path, f'{os.path.split(model_path)[-1]}_{best_step}_steps.zip'))
	# '''SAC-TFL'''
	# model = model_type.load(os.path.join(model_path, f'{os.path.split(model_path)[-1]}_{best_step}_steps'))
	# '''NAF2'''
	# model = model_type.load(os.path.join(model_path, f'step_{f"{best_step}".zfill(4)}'))

	# Plot the shit
	fig, axs = plt.subplots(2, NB_EPS, figsize=fig_size, gridspec_kw={'height_ratios': (4, 3)})
	fig.subplots_adjust(**tests_subplot_adjust_kwargs)
	axs = axs.flatten()
	fig.suptitle(title_name, size=title_font_size, weight=title_font_weight)

	prev_init_o = []
	for i in range(NB_EPS):
		while True:
			o = env.reset()
			if np.sqrt(np.sum(np.square(o))) < INIT_OBS_MIN_DIST:
				continue

			if len(prev_init_o) > 0:
				try:
					for _o in prev_init_o:
						if np.sqrt(np.sum(np.square(np.subtract(_o, o)))) < INIT_OBS_MIN_DIST:
							raise Exception
				except:
					continue
			else:
				break

			prev_init_o.append(o)
			break

		opt_env.reset(o.copy())
		opt_env.pi_controller_warm_up()

		states_model = o.copy().reshape((1, -1))
		states_pi = states_model.copy()
		actions_model = np.zeros(shape=(1, n_act))
		actions_pi = np.zeros_like(actions_model)
		rewards_model = []
		rewards_pi = []

		d_model = False
		d_pi = False

		for step in range(LIM_STEPS):  # env.EPISODE_LENGTH_LIMIT):
			if not d_model:
				a_model = model.predict(o.copy(), deterministic=True)[0] + noise()
				if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
					a_model = model.predict(o.copy(), deterministic=True) + noise()  # only for SACTFL
				a_model = np.clip(a_model, -1.0, 1.0)
				o, r, d_model, _ = env.step(a_model)
				actions_model = np.vstack([actions_model, a_model])
				states_model = np.vstack([states_model, o.copy()])
				rewards_model.append(r)

			if not d_pi:
				a_pi = opt_env.pi_controller_action() + noise()
				a_pi = np.clip(a_pi, -1.0, 1.0)
				o_pi, r_pi, d_pi, _ = opt_env.step(a_pi)
				actions_pi = np.vstack([actions_pi, a_pi])
				states_pi = np.vstack([states_pi, o_pi.copy()])
				rewards_pi.append(r_pi)

			if d_model and d_pi or step == LIM_STEPS - 1:
				axes = [axs[i], axs[i + NB_EPS]]

				ax = axes[0]
				ax.set_title(f'Episode #{i + 1}', size=subtitle_font_size, weight=subtitle_font_weight)
				ax.add_patch(plt.Circle((0, 0), radius=env.Q_goal * VISUAL_THRESH_MULTIPLIER,
				                        fill=False, edgecolor='g', ls='dashed',
				                        label='Threshold', lw=1, zorder=10))
				if i == 0:
					has_all_labels = True
				else:
					has_all_labels = False
				populate_ax_with_evaluation_episode_trace_2d_tune(axes, action_agent=actions_model,
				                                                  state_agent=states_model,
				                                                  action_pi=actions_pi, state_pi=states_pi,
				                                                  has_all_labels=has_all_labels)
				miny, maxy = ax.get_ylim()
				minx, maxx = ax.get_xlim()

				first_harmonic = (env.perturbation_center_freq // 50.0) * 50.0
				for k in np.arange(-5, 6, 1):
					harmonic_freq = first_harmonic + k * (50.0)
					harmonic_freq -= env.perturbation_center_freq
					harmonic_freq /= env.Q_LIMIT_HZ
					if miny <= harmonic_freq <= maxy:
						ax.axhline(harmonic_freq, color='k', ls='dashed', lw=1)
					if minx <= harmonic_freq <= maxx:
						ax.axvline(harmonic_freq, color='k', ls='dashed', lw=1)

				break

	_lines = [Line2D([0], [0], **kw) for kw in (agent_state_kw, pi_state_kw, agent_action_kw, pi_action_kw)]
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_start_color, marker='o', linewidth=episode_marker_lw))
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color, marker='o', linewidth=episode_marker_lw))
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color, marker='s', linewidth=episode_marker_lw))
	_lines.append(Patch(facecolor='w', edgecolor='g', ls='dashed', lw=1))
	_lines.append(Line2D([0], [0], color='k', ls='dashed', lw=1))

	for i in range(NB_EPS):
		miny, maxy = axs[i].get_ylim()
		miny = min(miny, -0.1)
		maxy = max(maxy, 0.1)
		minx, maxx = axs[i].get_xlim()
		minx = min(minx, -0.1)
		maxx = max(maxx, 0.1)

		axs[i].set_ylim((miny, maxy))
		axs[i].set_xlim((minx, maxx))

	fig.legend(_lines, ['Agent states', 'PI states', 'Agent actions', 'PI actions',
	                    'Episode Start', ' Agent episode end', 'PI episode end',
	                    f'State threshold x{VISUAL_THRESH_MULTIPLIER}', '50 Hz harmonic'], ncol=legend_ncol,
	           loc='lower center', frameon=False, prop={'size': legend_font_size,
	                                                    'weight': legend_font_weight})

	model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
	rl_type = model_name.split('_')[0]

	if save_dir is None:
		save_dir = os.path.join('results', 'policy_tests')
	elif not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_name = f'{model_name}_perturb-state'
	if action_noise_sigma != 0.0:
		save_name += f'_{action_noise_sigma * 100.0:.1f}-action-noise'

	file_cnt = 1
	while True:
		save_path = os.path.join(save_dir, f'{save_name}_{file_cnt}.pdf')
		if os.path.exists(save_path):
			file_cnt += 1
			continue
		else:
			break

	fig.savefig(save_path)
	print(f'Saved {model_path} to {save_path}')


def test_state_perturbation(model_paths, best_steps, name_parsers, model_types, env, opt_env,
                            action_noise_sigma=0.0):
	NB_EPS = 100
	INIT_OBS_MIN_DIST = 1 / 25
	LIM_STEPS = 70

	noise = lambda: np.random.normal(0.0, action_noise_sigma, env.act_dimension)

	n_obs = opt_env.obs_dimension
	n_act = opt_env.act_dimension

	mn_ep_len_agent_all = []
	mn_ep_len_pi_all = []

	for model_path, best_step, name_parser, model_type in zip(model_paths, best_steps, name_parsers, model_types):
		rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'
		model_name = os.path.split(model_path)[-1]

		full_model_path = os.path.join(model_path, name_parser(model_name, best_step))
		model = model_type.load(full_model_path)
		if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
			model.policy_net(env.reset().reshape((1, -1)).astype(np.float32))  # only for SAC-TFl
		print(f'-> Loaded {full_model_path}')

		prev_init_o = []
		ep_len_agent = []
		ep_len_pi = []
		for i in pbar(range(NB_EPS)):
			o = env.reset()
			# Crazy episode initialisation scheme
			# while True:
			#     o = env.reset()
			#     if np.sqrt(np.sum(np.square(o))) < INIT_OBS_MIN_DIST:
			#         continue
			#
			#     if len(prev_init_o) > 0:
			#         try:
			#             for _o in prev_init_o:
			#                 if np.sqrt(np.sum(np.square(np.subtract(_o, o)))) < INIT_OBS_MIN_DIST:
			#                     raise Exception
			#         except:
			#             continue
			#     else:
			#         break
			#
			#     prev_init_o.append(o)
			#     break

			opt_env.reset(o.copy())
			opt_env.pi_controller_warm_up()

			d_model = False
			d_pi = False

			for step in range(LIM_STEPS):  # env.EPISODE_LENGTH_LIMIT):
				if not d_model:
					a_model = model.predict(o.copy(), deterministic=True)[0] + noise()
					if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
						a_model = model.predict(o.copy(), deterministic=True) + noise()  # only for SACTFL
					a_model = np.clip(a_model, -1.0, 1.0)
					o, r, d_model, _ = env.step(a_model)
					step_agent_finish = copy(step)

				if not d_pi:
					a_pi = opt_env.pi_controller_action() + noise()
					a_pi = np.clip(a_pi, -1.0, 1.0)
					o_pi, r_pi, d_pi, _ = opt_env.step(a_pi)
					step_pi_finish = copy(step)

				if d_model and d_pi or step == LIM_STEPS - 1:
					ep_len_agent.append(step_agent_finish)
					ep_len_pi.append(step_pi_finish)
					i += 1
					break

		mn_ep_len_agent = np.mean(ep_len_agent)
		mn_ep_len_pi = np.mean(ep_len_pi)

		mn_ep_len_agent_all.append(mn_ep_len_agent)
		mn_ep_len_pi_all.append(mn_ep_len_pi)

		with open(os.path.join('results', 'policy_tests', 'perturb_state',
		                       f'info_agents_ep_lens_{action_noise_sigma * 100:.1f}-action-noise.txt'), 'a') as f:
			f.write(f'-> {model_path} @ step {best_step}\n')
			f.write(f'\t`-> Agent mean episode length: {mn_ep_len_agent}\n')
			f.write(f'\t`-> PI mean episode length: {mn_ep_len_pi}\n\n\n')

	agent_idx = np.argmin(mn_ep_len_agent_all)
	pi_idx = np.argmin(mn_ep_len_pi_all)

	print(f' -> Best agent: {model_paths[agent_idx]} Ep. len: {mn_ep_len_agent_all[agent_idx]}')
	print(f' -> Best PI Ep. len: {mn_ep_len_pi_all[pi_idx]}')


'''
ACTION FAILURE TEST
'''


def save_episode_trace_only_best_act_failure(model_path, best_step, name_parser, model_type, env, opt_env, env2,
                                             opt_env2, title_name,
                                             action_noise_sigma=0.0, save_dir=None):
	NB_EPS = 3
	INIT_OBS_MIN_DIST = 0.3
	LIM_STEPS = 20
	VISUAL_THRESH_MULTIPLIER = 2
	MAX_FAILURES = 5

	# NB_FAILURES = 1
	step_to_fail = np.arange(1000)  # * NB_FAILURES
	for NB_FAILURES in np.arange(1, MAX_FAILURES + 1, 2):
		print( '********************************')
		print(f'Failure test: Nb. failures = {NB_FAILURES}')
		n_act = env.act_dimension
		noise = lambda: np.random.normal(0.0, action_noise_sigma, env.act_dimension)

		rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'
		model_name = os.path.split(model_path)[-1]

		# if 'AE-DYNA' in rl_type:
		#     model = model_type.load(os.path.join(model_path, 'agent', name_parser(model_path, best_step)))
		#     model.policy_net(env.reset().reshape((1, -1)).astype(np.float32)) # only for SACTFl
		#
		# else:
		#     model = model_type.load(os.path.join(model_path, name_parser(model_name, best_step)))
		#     if 'SAC-TFL' in rl_type:
		#         model.policy_net(env.reset().reshape((1, -1)).astype(np.float32)) # only for SACTFl
		full_model_path = os.path.join(model_path, name_parser(model_path, best_step))
		model = model_type.load(full_model_path)
		if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
			model.policy_net(env.reset().reshape((1, -1)).astype(np.float32))  # only for SAC-TFl

		# '''stable-baselines'''
		# model = model_type.load(os.path.join(model_path, f'{os.path.split(model_path)[-1]}_{best_step}_steps.zip'))
		# '''SAC-TFL'''
		# model = model_type.load(os.path.join(model_path, f'{os.path.split(model_path)[-1]}_{best_step}_steps'))
		# '''NAF2'''
		# model = model_type.load(os.path.join(model_path, f'step_{f"{best_step}".zfill(4)}'))

		# Plot the shit
		fig, axs = plt.subplots(2, NB_EPS, figsize=actfail_fig_size, gridspec_kw={'height_ratios': (4, 3)})
		fig.subplots_adjust(**actfail_tests_subplot_adjust_kwargs)
		axs = axs.flatten()
		actual_title_name = f'{title_name} = {NB_FAILURES}'
		fig.suptitle(actual_title_name, size=title_font_size, weight=title_font_weight)

		prev_init_o = []
		prev_init_failure = []
		for i in range(NB_EPS):
			# both act_indices and step_to_fail are indexed by the same action
			fail_indices = []
			mask = np.ones(n_act)

			def augment_action(cur_step, act):
				nonlocal mask, prev_init_failure

				'''Return the same masked action or find a new action to fail'''
				if cur_step >= step_to_fail[len(fail_indices)] and len(fail_indices) < NB_FAILURES:
					print(f'Episode #{i} -> looking for a failure')
					while True:
						act_idx = np.random.choice(n_act)
						'''Make sure that the action being turned off is not -1, and that there are enough non-negone actions 
						to choose from'''
						if (act_idx in fail_indices) or \
								(act[act_idx] < -0.9 and sum(
									np.where(act < -0.9, np.zeros_like(act), np.ones_like(act))) > len(fail_indices)):
							print(f'\t Failure at index {act_idx} skipped: Negone alert')
							continue
						elif cur_step == 0 and act_idx in prev_init_failure:
							'''if this failure happened at the start of the other episodes, skip it'''
							print(f'\t Failure at index {act_idx} skipped: Happened before')
							continue
						else:
							print(f'\t Failure at index {act_idx}: ACCEPTED')
							if cur_step == 0:
								prev_init_failure.append(act_idx)
							fail_indices.append(act_idx)
							mask[act_idx] = 0
							print(f'found one at index {act_idx} value={act[act_idx]}')
							break
				# if cur_step == step_to_fail[len(fail_indices) - 1]:
				# 	mask[fail_indices[-1]] = 0
				# break
				return np.array([item if mask[j] else -1 for j, item in enumerate(act)])

			# Ensure that all episodes are substantially different
			while True:
				o = env.reset()
				o2 = copy(o)
				if np.sqrt(np.sum(np.square(o))) < INIT_OBS_MIN_DIST:
					continue

				if len(prev_init_o) > 0:
					try:
						for _o in prev_init_o:
							if np.sqrt(np.sum(np.square(np.subtract(_o, o)))) < INIT_OBS_MIN_DIST:
								raise Exception
					except:
						continue

				prev_init_o.append(o)
				break

			# Initialise PI controller environment with same state
			env2.reset(o2.copy())
			opt_env.reset(o.copy())
			opt_env2.reset(o2.copy())
			opt_env.pi_controller_warm_up()
			opt_env2.pi_controller_warm_up()

			# Collecting episode trajectory
			states_model = o.copy().reshape((1, -1))
			states_pi = states_model.copy()
			states_model2 = o.copy().reshape((1, -1))
			states_pi2 = states_model2.copy()
			actions_model = np.zeros(shape=(1, n_act))
			actions_pi = np.zeros_like(actions_model)
			rewards_model = []
			rewards_pi = []

			d_model = False
			d_model2 = False
			d_pi = False
			d_pi2 = False

			# Start episode
			for step in range(LIM_STEPS):  # env.EPISODE_LENGTH_LIMIT):
				if not d_model:
					if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
						a_model = model.predict(o.copy(), deterministic=True) + noise()  # only for SACTFL
					else:
						a_model = model.predict(o.copy(), deterministic=True)[0] + noise()

					a_model = np.clip(a_model, -1.0, 1.0)
					a_model = augment_action(step, np.copy(a_model))

					o, r, d_model, _ = env.step(a_model)
					actions_model = np.vstack([actions_model, a_model])
					states_model = np.vstack([states_model, o.copy()])
					rewards_model.append(r)

				if not d_model2:
					if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
						a_model = model.predict(o2.copy(), deterministic=True) + noise()  # only for SACTFL
					else:
						a_model = model.predict(o2.copy(), deterministic=True)[0] + noise()
					a_model = np.clip(a_model, -1.0, 1.0)
					o2, _, d_model2, _ = env2.step(a_model)
					states_model2 = np.vstack([states_model2, o2.copy()])

				if not d_pi:
					a_pi = opt_env.pi_controller_action() + noise()
					a_pi = np.clip(a_pi, -1.0, 1.0)

					a_pi = augment_action(step, np.copy(a_pi))

					o_pi, r_pi, d_pi, _ = opt_env.step(a_pi)
					actions_pi = np.vstack([actions_pi, a_pi])
					states_pi = np.vstack([states_pi, o_pi.copy()])
					rewards_pi.append(r_pi)

				if not d_pi2:
					a_pi = opt_env2.pi_controller_action() + noise()
					a_pi = np.clip(a_pi, -1.0, 1.0)
					o_pi2, _, d_pi2, _ = opt_env2.step(a_pi)
					states_pi2 = np.vstack([states_pi2, o_pi2.copy()])

				if d_model and d_pi or step == LIM_STEPS - 1:
					axes = [axs[i], axs[i + NB_EPS]]

					ax = axes[0]

					for s, c in zip(f'',
					                (actfail_color, 'r', actfail_color)):
						text = ax.text(0, 0, s + " ", color=c, transform=t, size=subtitle_font_size,
						               weight=subtitle_font_weight)
						text.draw(fig.canvas.get_renderer())
						ex = text.get_window_extent()
						t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

					ax.set_title(
						f'Episode #{i + 1}\nAct. Fail index: {fail_indices}\nAt step: {step_to_fail[:NB_FAILURES]}',
						color=actfail_color, size=actfail_subtitle_font_size, weight=actfail_subtitle_font_weight)

					ax.add_patch(plt.Circle((0, 0), radius=env.Q_goal * VISUAL_THRESH_MULTIPLIER,
					                        fill=False, edgecolor='g', ls=threshold_style,
					                        label='Threshold', lw=threshold_lw, zorder=10))

					if i == 0:
						has_all_labels = True
					else:
						has_all_labels = False
					populate_ax_with_evaluation_episode_trace_2d_tune_highlight_failure(axes,
					                                                                    action_agent=actions_model,
					                                                                    state_agent=states_model,
					                                                                    state_agent2=states_model2,
					                                                                    action_pi=actions_pi,
					                                                                    state_pi=states_pi,
					                                                                    state_pi2=states_pi2,
					                                                                    fail_indices=fail_indices,
					                                                                    has_all_labels=has_all_labels)

					break

		# Set state axis limits to be a bit bigger than data shown
		for i in range(NB_EPS):
			miny, maxy = axs[i].get_ylim()
			miny = min(miny, -0.1)
			maxy = max(maxy, 0.1)
			minx, maxx = axs[i].get_xlim()
			minx = min(minx, -0.1)
			maxx = max(maxx, 0.1)

			axs[i].set_ylim((miny, maxy))
			axs[i].set_xlim((minx, maxx))

		# Lines for legend
		_lines = [Line2D([0], [0], **kw) for kw in (agent_state2_kw, agent_state_kw, pi_state2_kw, pi_state_kw,
		                                            agent_action_kw, agent_actfail_kw, pi_action_kw,
		                                            pi_actfail_kw)]
		_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_start_color,
		                     marker='o', markeredgewidth=episode_marker_lw))
		_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color,
		                     marker='o', markeredgewidth=episode_marker_lw))
		_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color,
		                     marker='s', markeredgewidth=episode_marker_lw))
		_lines.append(Patch(facecolor='w', edgecolor='g', ls=threshold_style, lw=threshold_lw))

		fig.legend(_lines,
		           ['Agent states (before)', 'Agent states (after)', 'PI states (before)', 'PI states (after)',
		            'Agent actions', 'Agent failure', 'PI actions', 'PI failure',
		            'Episode start', ' Agent episode end', 'PI episode end',
		            f'State threshold x{VISUAL_THRESH_MULTIPLIER}'],
		           ncol=actfail_legend_ncol, loc='lower center', frameon=False, prop={'size': legend_font_size,
		                                                                              'weight': legend_font_weight})

		# Create save directory
		if save_dir is None:
			save_dir = os.path.join('results', 'policy_tests')
		elif not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# Create save name
		save_name = f'{model_name}_{NB_FAILURES}_act_failure'
		if action_noise_sigma != 0.0:
			save_name += f'_{action_noise_sigma * 100.0:.1f}-action-noise'

		# Update file name index to avoid overwriting
		file_cnt = 1
		while True:
			save_path = os.path.join(save_dir, f'{save_name}_{file_cnt}.pdf')
			if os.path.exists(save_path):
				file_cnt += 1
				continue
			else:
				break

		fig.savefig(save_path)
		print(f'Saved {model_path} to {save_path}')
	# plt.show()


'''
GAUSSIAN ACTION NOISE TEST
'''


def save_episode_trace_only_best_step_action_noise(model_path, best_step, name_parser, model_type,
                                                   env, opt_env, title_name, action_noise_sigma=0.0,
                                                   save_dir=None):
	NB_EPS = 3
	INIT_OBS_MIN_DIST = 0.4

	noise = lambda: np.random.normal(0.0, action_noise_sigma, env.act_dimension)

	n_act = opt_env.act_dimension

	rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'
	model_name = os.path.split(model_path)[-1]

	temp = os.path.join(model_path, name_parser(model_name, best_step))
	model = model_type.load(temp)
	if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
		model.policy_net(env.reset().reshape((1, -1)).astype(np.float32))  # only for SACTFl

	# '''stable-baselines'''
	# model = model_type.load(os.path.join(model_path, f'{os.path.split(model_path)[-1]}_{best_step}_steps.zip'))
	# '''SAC-TFL'''
	# model = model_type.load(os.path.join(model_path, f'{os.path.split(model_path)[-1]}_{best_step}_steps'))
	# '''NAF2'''
	# model = model_type.load(os.path.join(model_path, f'step_{f"{best_step}".zfill(4)}'))

	# Plot the shit
	fig, axs = plt.subplots(2, NB_EPS, figsize=fig_size, gridspec_kw={'height_ratios': (4, 3)})
	fig.subplots_adjust(**tests_subplot_adjust_kwargs)
	axs = axs.flatten()
	fig.suptitle(title_name, size=title_font_size, weight=title_font_weight)

	prev_init_o = []
	for i in range(NB_EPS):
		while True:
			o = env.reset()
			if np.sqrt(np.sum(np.square(o))) < INIT_OBS_MIN_DIST:
				continue

			if len(prev_init_o) > 0:
				try:
					for _o in prev_init_o:
						if np.sqrt(np.sum(np.square(np.subtract(_o, o)))) < INIT_OBS_MIN_DIST:
							raise Exception
				except:
					continue
			else:
				break

			prev_init_o.append(o)
			break

		opt_env.reset(o.copy())
		opt_env.pi_controller_warm_up()

		states_model = o.copy().reshape((1, -1))
		states_pi = states_model.copy()
		actions_model = np.zeros(shape=(1, n_act))
		actions_pi = np.zeros_like(actions_model)
		rewards_model = []
		rewards_pi = []

		d_model = False
		d_pi = False

		for step in range(env.EPISODE_LENGTH_LIMIT):
			if not d_model:
				a_model = model.predict(o.copy(), deterministic=True)[0] + noise()
				if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
					a_model = model.predict(o.copy(), deterministic=True) + noise()  # only for SACTFL
				a_model = np.clip(a_model, -1.0, 1.0)
				o, r, d_model, _ = env.step(a_model)
				actions_model = np.vstack([actions_model, a_model])
				states_model = np.vstack([states_model, o.copy()])
				rewards_model.append(r)

			if not d_pi:
				a_pi = opt_env.pi_controller_action() + noise()
				a_pi = np.clip(a_pi, -1.0, 1.0)
				o_pi, r_pi, d_pi, _ = opt_env.step(a_pi)
				actions_pi = np.vstack([actions_pi, a_pi])
				states_pi = np.vstack([states_pi, o_pi.copy()])
				rewards_pi.append(r_pi)

			if d_model and d_pi:
				axes = [axs[i], axs[i + NB_EPS]]

				ax = axes[0]
				ax.set_title(f'Episode #{i + 1}', size=subtitle_font_size, weight=subtitle_font_weight)
				ax.add_patch(plt.Circle((0, 0), radius=env.Q_goal,
				                        fill=False, edgecolor='g', ls='dashed',
				                        label='Threshold', lw=1, zorder=10))

				if i == 0:
					has_all_labels = True
				else:
					has_all_labels = False
				populate_ax_with_evaluation_episode_trace_2d_tune(axes, action_agent=actions_model,
				                                                  state_agent=states_model,
				                                                  action_pi=actions_pi, state_pi=states_pi,
				                                                  has_all_labels=has_all_labels)
				i += 1
				break

	_lines = [Line2D([0], [0], **kw) for kw in (agent_state_kw, pi_state_kw, agent_action_kw, pi_action_kw)]
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_start_color, marker='o', linewidth=episode_marker_lw))
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color, marker='o', linewidth=episode_marker_lw))
	_lines.append(Line2D([0], [0], color='None', markerfacecolor='None', markeredgecolor=episode_end_color, marker='s', linewidth=episode_marker_lw))
	_lines.append(Patch(facecolor='w', edgecolor='g', ls=threshold_style, lw=threshold_lw))

	fig.legend(_lines, ['Agent states', 'PI states', 'Agent actions', 'PI actions',
	                    'Episode start', ' Agent episode end', 'PI episode end',
	                    'State threshold'], ncol=legend_ncol, loc='lower center', frameon=False,
	           prop={'size': legend_font_size,
	                 'weight': legend_font_weight})

	model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
	rl_type = model_name.split('_')[0]
	if action_noise_sigma == 0.0:
		noise_name = 'no-action-noise'
	else:
		noise_name = f'{action_noise_sigma * 100.0:.1f}-action-noise'

	if save_dir is None:
		save_dir = os.path.join('results', 'policy_tests')
	elif not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_name = f'{model_name}_{noise_name}'

	file_cnt = 1
	while True:
		save_path = os.path.join(save_dir, f'{save_name}_{file_cnt}.ps')
		if os.path.exists(save_path):
			file_cnt += 1
			continue
		else:
			break

	fig.savefig(save_path)
	print(f'Saved {model_path} to {save_path}')


def calculate_episode_statistics_best_step_action_noise(model_path, best_step, name_parser, model_type,
												  env, opt_env):
	NB_EPS = 1000
	NB_INTERVALS = 21

	n_obs = env.obs_dimension
	n_act = env.act_dimension

	rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'
	model_name = os.path.split(model_path)[-1]

	temp = os.path.join(model_path, name_parser(model_name, best_step))
	model = model_type.load(temp)
	if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
		model.policy_net(env.reset().reshape((1, -1)).astype(np.float32))  # only for SACTFl

	init_o = np.random.uniform(-1, 1, NB_EPS * n_obs).reshape((-1, n_obs))

	ep_lens_agent = np.zeros(shape=(NB_INTERVALS, NB_EPS))
	ep_lens_pi = np.zeros_like(ep_lens_agent)

	dist_to_center_agent = np.zeros_like(ep_lens_agent)
	dist_to_center_pi = np.zeros_like(ep_lens_agent)

	final_a_agent = np.zeros(shape=(NB_INTERVALS, NB_EPS, n_act))
	final_a_pi = np.zeros_like(final_a_agent)

	action_noise_sigmas = np.linspace(0.0, 0.5, NB_INTERVALS)
	'''Evaluate performance with varying action noise'''
	for i, action_noise_sigma in tqdm(enumerate(action_noise_sigmas)):
		noise = lambda: np.random.normal(0.0, action_noise_sigma, n_act)

		for j in range(NB_EPS):
			o = np.copy(init_o[j])
			env.reset(o)
			opt_env.reset(o.copy())
			opt_env.pi_controller_warm_up()

			d_model, d_pi = False, False
			for step in range(env.EPISODE_LENGTH_LIMIT):
				if not d_model:
					if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
						a_model = model.predict(o.copy(), deterministic=True) + noise()
					else:
						a_model = model.predict(o.copy(), deterministic=True)[0] + noise()
					a_model = np.clip(a_model, -1.0, 1.0)
					o, r, d_model, _  = env.step(a_model)
				else:
					break

			ep_lens_agent[i, j] = step + 1

			dist = np.sqrt(np.sum(np.square(o)))
			dist_to_center_agent[i, j] = dist

			final_a_agent[i, j] = a_model

			for step in range(env.EPISODE_LENGTH_LIMIT):
				if not d_pi:
					a_pi = opt_env.pi_controller_action() + noise()
					a_pi = np.clip(a_pi, -1.0, 1.0)
					o_pi, r_pi, d_pi, _  = opt_env.step(a_pi)
				else:
					break

			ep_lens_pi[i, j] = step + 1

			dist = np.sqrt(np.sum(np.square(o_pi)))
			dist_to_center_pi[i, j] = dist

			final_a_pi[i, j] = a_pi

	return dict(action_noise_sigmas=action_noise_sigmas,
				ep_lens_agent=ep_lens_agent, ep_lens_pi=ep_lens_pi,
				dist_to_center_agent=dist_to_center_agent, dist_to_center_pi=dist_to_center_pi,
				final_a_agent=final_a_agent, final_a_pi=final_a_pi)

def plot_episode_statistics_best_step_action_noise(model_path, best_step, name_parser, model_type, env, opt_env):
	rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'

	data = calculate_episode_statistics_best_step_action_noise(model_path, best_step, name_parser, model_type, env, opt_env)
	action_noise_sigmas = data['action_noise_sigmas']
	ep_lens_agent = data['ep_lens_agent']
	ep_lens_pi = data['ep_lens_pi']
	dist_to_center_agent = data['dist_to_center_agent']
	dist_to_center_pi = data['dist_to_center_pi']
	final_a_agent = data['final_a_agent']
	final_a_pi = data['final_a_pi']

	get_quartiles = lambda arr: df(arr).quantile([0.0, 0.25, 0.5, 0.75, 1.0], axis=1, interpolation='nearest').to_numpy()

	def plot_quartiles(xdata, agent_data, pi_data, title_name, xlabel, ylabel):
		fig, ax = plt.subplots(figsize=(10, 7))
		fig.suptitle(title_name, size=18)

		ax.plot(xdata, agent_data[2], c='b')
		ax.plot(xdata, agent_data[0], c='b', ls=':')
		ax.plot(xdata, agent_data[4], c='b', ls=':')
		ax.fill_between(xdata, agent_data[1], agent_data[3], color='b', alpha=0.4, hatch='\\')

		ax.plot(xdata, pi_data[2], c='r')
		ax.plot(xdata, pi_data[0], c='r', ls=':')
		ax.plot(xdata, pi_data[4], c='r', ls=':')
		ax.fill_between(xdata, pi_data[1], pi_data[3], color='r', alpha=0.4, hatch='\\')

		legend_handles = [Line2D([0], [0], c='b'),
						Line2D([0], [0], c='r'),
						Patch(facecolor='gray', alpha=0.4, hatch='\\'),
						Line2D([0], [0], c='gray', ls=':')]
		legend_labels = ['Agent Median', 'PI controller Median', '1st - 3rd quartile', 'Lower - Upper bounds']
		ax.legend(legend_handles, legend_labels, loc='upper left', prop={'size':15})
		ax.set_ylabel(ylabel, size=15)
		ax.set_xlabel(xlabel, size=15)
		ax.grid()
		fig.tight_layout()

		return ax, legend_handles, legend_labels

	get_savepath = lambda hajta: os.path.join('results', 'episode-statistics', f'{rl_type.lower()}_{hajta}_vs_action-noise-{NB_INTERVALS}-intervals_{NB_EPS}-episodes.png')
	# def get_savepath()
	'''Episode length stats'''
	ela = get_quartiles(ep_lens_agent)
	elp = get_quartiles(ep_lens_pi)

	for table_action_noise in (0.0, 0.1, 0.25, 0.5):
		if table_action_noise in action_noise_sigmas:
			idx = np.searchsorted(action_noise_sigmas, table_action_noise)
			print(f'{rl_type}')
			print(f'-> Action noise sigma = {action_noise_sigmas[idx]}')
			print(f'\t-> Agent Mean ep.len. = {np.mean(ep_lens_agent[idx]):.4f}')
			print(f'\t-> Agent Std  ep.len. = {np.std(ep_lens_agent[idx]):.4f}')
			print(f'\t-> Agent Mean dtc = {np.mean(dist_to_center_agent[idx]):.4f}')
			print(f'\t-> Agent Std  dtc = {np.std(dist_to_center_agent[idx]):.4f}')
			print(f'\t-> PI Mean ep.len. = {np.mean(ep_lens_pi[idx]):.4f}')
			print(f'\t-> PI Std  ep.len. = {np.std(ep_lens_pi[idx]):.4f}')
			print(f'\t-> PI Mean dtc = {np.mean(dist_to_center_pi[idx]):.4f}')
			print(f'\t-> PI Std  dtc = {np.std(dist_to_center_pi[idx]):.4f}')

	action_noise_sigmas *= 100.0

	title_name = f'{rl_type} Best policy evaluation\nEpisode length vs. Action noise\nNb. episodes = {NB_EPS}'
	xlabel = 'Action noise (%)'
	ylabel = 'Episode length'

	ax, *_ = plot_quartiles(action_noise_sigmas, ela, elp, title_name, xlabel, ylabel)
	ax.set_ylim((-3, env.EPISODE_LENGTH_LIMIT + 3))

	plt.savefig(get_savepath('ep-len'))


	'''Distance to center (optimal point) stats'''
	dtca = get_quartiles(dist_to_center_agent)
	dtcp = get_quartiles(dist_to_center_pi)

	title_name = f'{rl_type} Best policy evaluation\nFinal distance to optimal point vs. Action noise\nNb. episodes = {NB_EPS}'
	ylabel = '$\sqrt{\Delta Q_H^2 + \Delta Q_V^2}$'

	ax, handles, labels = plot_quartiles(action_noise_sigmas, dtca, dtcp, title_name, xlabel, ylabel)
	ax.get_legend().remove()

	goal_line = np.sqrt(np.sum(np.square([env.Q_goal, env.Q_goal])))
	ax.axhline(goal_line, color='g', ls='--')

	handles.append(Line2D([0], [0], color='g', ls='--'))
	labels.append('Goal threshold')
	ax.legend(handles, labels, prop=dict(size=15), loc='best')

	plt.savefig(get_savepath('dtc'))

	'''Action values at episode end'''
	faa = get_quartiles(final_a_agent.reshape((NB_INTERVALS, -1)))
	fap = get_quartiles(final_a_pi.reshape((NB_INTERVALS, -1)))
	title_name = f'{rl_type} Best policy evaluation\nAction values at episode end vs. Action noise\nNb. episodes = {NB_EPS}'
	ylabel = 'Action values (normalised)'
	ax, *_ = plot_quartiles(action_noise_sigmas, faa, fap, title_name, xlabel, ylabel)
	# ax.set_yscale('symlog', linthresh=0.1, linscale=2)
	ax.grid(which='major', axis='y', color='gray')
	ax.minorticks_on()
	ax.grid(which='minor', axis='y', color='gray', linestyle='--', alpha=0.5)

	plt.savefig(get_savepath('act-vals'))


def frontiers_merge_high_stats_episode_statistics():
	FIG_SIZE = (12, 5)
	NB_INTERVALS = 21
	NB_EPS = 1000
	LEGEND_FS = 13
	LEGEND_ALPHA = 0.3
	filenames = [f'naf2_save-episode-statistics_{NB_INTERVALS}-intervals_{NB_EPS}-episodes.pkl',
				 f'ppo_save-episode-statistics_{NB_INTERVALS}-intervals_{NB_EPS}-episodes.pkl']

	data = {}
	for filename in filenames:
		with open(filename, 'rb') as f:
			rl_type = filename.split('_')[0]
			data[rl_type] = pkl.load(f)

	get_quartiles = lambda arr: df(arr).quantile([0.0, 0.25, 0.5, 0.75, 1.0], axis=1,
												 interpolation='nearest').to_numpy()

	def plot_quartiles(xdata, agent_data, pi_data, title_name, xlabel, ylabel, ax=None, rl_type='Agent'):
		_ax = None
		if ax is None:
			fig, ax = plt.subplots(figsize=(10, 7))
			fig.suptitle(title_name, size=18)
		else:
			_ax = ax

		ax.plot(xdata, agent_data[2], c='b')
		ax.plot(xdata, agent_data[0], c='b', ls=':')
		ax.plot(xdata, agent_data[4], c='b', ls=':')
		ax.fill_between(xdata, agent_data[1], agent_data[3], color='b', alpha=0.4, hatch='\\')

		ax.plot(xdata, pi_data[2], c='r')
		ax.plot(xdata, pi_data[0], c='r', ls=':')
		ax.plot(xdata, pi_data[4], c='r', ls=':')
		ax.fill_between(xdata, pi_data[1], pi_data[3], color='r', alpha=0.4, hatch='\\')

		legend_handles = [Line2D([0], [0], c='b'),
						  Line2D([0], [0], c='r'),
						  Patch(facecolor='gray', alpha=0.4, hatch='\\'),
						  Line2D([0], [0], c='gray', ls=':')]
		legend_labels = [f'{rl_type.upper()} Median', 'PI controller Median', '1st - 3rd quartile', 'Lower - Upper bounds']
		leg = ax.legend(legend_handles, legend_labels, loc='upper left', prop={'size': LEGEND_FS})
		for lh in leg.legendHandles:
			lh.set_alpha(LEGEND_ALPHA)
		ax.set_ylabel(ylabel, size=15)
		ax.set_xlabel(xlabel, size=15)
		ax.grid()
		try:
			fig.tight_layout()
		except:
			pass

		return ax, legend_handles, legend_labels

	get_savepath = lambda hajta: os.path.join('results', 'frontiers',
											  f'naf2ppo_{hajta}_vs_action-noise-{NB_INTERVALS}-intervals_{NB_EPS}-episodes.png')
	# def get_savepath()
	'''Episode length stats'''
	elppo = get_quartiles(data['ppo']['ep_lens_agent'])
	elp1 = get_quartiles(data['ppo']['ep_lens_pi'])
	elnaf2 = get_quartiles(data['naf2']['ep_lens_agent'])
	elp2 = get_quartiles(data['naf2']['ep_lens_pi'])

	action_noise_sigmas = data['ppo']['action_noise_sigmas']
	action_noise_sigmas *= 100.0

	title_name = f'Episode length vs. Action noise\nNb. episodes = {NB_EPS}'
	xlabel = 'Action noise (%)'
	ylabel = 'Episode length'

	fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE)
	plot_quartiles(action_noise_sigmas, elnaf2, elp2, title_name, xlabel, ylabel, axs[0], rl_type='NAF2')
	plot_quartiles(action_noise_sigmas, elppo, elp1, title_name, xlabel, ylabel, axs[1], rl_type='PPO')
	for ax in axs:
		ax.set_ylim((-3, 73))
	fig.tight_layout()

	plt.savefig(get_savepath('ep-len'))

	'''Distance to center (optimal point) stats'''
	dtoppo = get_quartiles(data['ppo']['dist_to_center_agent'])
	dtop1 = get_quartiles(data['ppo']['dist_to_center_pi'])
	dtonaf2 = get_quartiles(data['naf2']['dist_to_center_agent'])
	dtop2 = get_quartiles(data['naf2']['dist_to_center_pi'])

	title_name = f'Final distance to optimal point vs. Action noise\nNb. episodes = {NB_EPS}'
	ylabel = '$\sqrt{\Delta Q_H^2 + \Delta Q_V^2}$'

	fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE)
	for ax, agent_data, pi_data, rl_type in zip(axs, (dtonaf2, dtoppo), (dtop2, dtop1), ('naf2', 'ppo')):
		_, handles, labels = plot_quartiles(action_noise_sigmas, agent_data, pi_data, title_name, xlabel, ylabel, ax, rl_type=rl_type)
		ax.get_legend().remove()

		goal_line = np.sqrt(np.sum(np.square([1/25, 1/25])))
		ax.axhline(goal_line, color='g', ls='--')

		handles.append(Line2D([0], [0], color='g', ls='--'))
		labels.append('Goal threshold')
		leg = ax.legend(handles, labels, prop=dict(size=LEGEND_FS), loc='best')
		for lh in leg.legendHandles:
			lh.set_alpha(LEGEND_ALPHA)
	fig.tight_layout()
	plt.savefig(get_savepath('dtc'))

	'''Action values at episode end'''
	fappo = get_quartiles(data['ppo']['final_a_agent'].reshape((NB_INTERVALS, -1)))
	fap1 = get_quartiles(data['ppo']['final_a_pi'].reshape((NB_INTERVALS, -1)))
	fanaf2 = get_quartiles(data['naf2']['final_a_agent'].reshape((NB_INTERVALS, -1)))
	fap2 = get_quartiles(data['naf2']['final_a_pi'].reshape((NB_INTERVALS, -1)))

	title_name = f'Action values at episode end vs. Action noise\nNb. episodes = {NB_EPS}'
	ylabel = 'Action values (normalised)'

	fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE)
	plot_quartiles(action_noise_sigmas, fanaf2, fap2, title_name, xlabel, ylabel, ax=axs[0], rl_type='NAF2')
	plot_quartiles(action_noise_sigmas, fappo, fap1, title_name, xlabel, ylabel, ax=axs[1], rl_type='PPO')
	# ax.set_yscale('symlog', linthresh=0.1, linscale=2)
	for ax in axs:
		ax.grid(which='major', axis='y', color='gray')
		ax.minorticks_on()
		ax.grid(which='minor', axis='y', color='gray', linestyle='--', alpha=0.5)
	fig.tight_layout()
	plt.savefig(get_savepath('act-vals'))






def get_episode_statistics_best_step_actuator_failure(model_path, best_step, name_parser, model_type,
												  env, opt_env):
	NB_EPS = 1000
	MAX_NB_FAILURES = 5

	n_obs = env.obs_dimension
	n_act = env.act_dimension

	rl_type = f'{os.path.split(model_path)[-1].split("_")[0]}'
	model_name = os.path.split(model_path)[-1]

	temp = os.path.join(model_path, name_parser(model_name, best_step))
	model = model_type.load(temp)
	if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
		model.policy_net(env.reset().reshape((1, -1)).astype(np.float32))  # only for SACTFl

	init_o = np.random.uniform(-1, 1, NB_EPS * n_obs).reshape((-1, n_obs))

	ep_lens_agent = np.zeros(shape=(MAX_NB_FAILURES + 1, NB_EPS))
	ep_lens_pi = np.zeros_like(ep_lens_agent)

	dist_to_center_agent = np.zeros_like(ep_lens_agent)
	dist_to_center_pi = np.zeros_like(ep_lens_agent)

	# final_a_agent = np.zeros(shape=(NB_INTERVALS, NB_EPS, n_act))
	# final_a_pi = np.zeros_like(final_a_agent)

	range_failures = np.arange(0, MAX_NB_FAILURES + 1)
	step_to_fail = np.arange(1000)
	'''Evaluate performance with varying action noise'''
	for i, nb_failures in tqdm(enumerate(range_failures)):
		for j in range(NB_EPS):
			fail_indices = []
			mask = np.ones(n_act)

			def augment_action(cur_step, act):
				nonlocal mask, nb_failures, fail_indices

				'''Return the same masked action or find a new action to fail'''
				if cur_step >= step_to_fail[len(fail_indices)] and len(fail_indices) < nb_failures:
					# print(f'Episode #{j} -> looking for a failure')
					while True:
						act_idx = np.random.choice(n_act)
						'''Make sure that the action being turned off is not -1, and that there are enough non-negone actions 
                        to choose from'''
						if (act_idx in fail_indices):
							# print(f'\t Failure at index {act_idx}: SKIP Already failed')
							continue

						elif (act[act_idx] < -0.9 and sum(np.where(act < -0.9,
																   np.zeros_like(act),
																   np.ones_like(act))) > len(fail_indices)):
							# print(f'\t Failure at index {act_idx}: SKIP Negone alert')
							continue
						else:
							# print(f'\t Failure at index {act_idx}: ACCEPTED')
							fail_indices.append(act_idx)
							mask[act_idx] = 0
							# print(f'found one at index {act_idx} value={act[act_idx]}')
							break
				# if cur_step == step_to_fail[len(fail_indices) - 1]:
				# 	mask[fail_indices[-1]] = 0
				# break
				return np.array([item if mask[j] else -1 for j, item in enumerate(act)])

			o = np.copy(init_o[j])
			env.reset(o)
			opt_env.reset(o.copy())
			opt_env.pi_controller_warm_up()

			d_model, d_pi = False, False
			for step in range(env.EPISODE_LENGTH_LIMIT):
				if not d_model:
					if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
						a_model = model.predict(o.copy(), deterministic=True)
					else:
						a_model = model.predict(o.copy(), deterministic=True)[0]
					a_model = np.clip(a_model, -1.0, 1.0)
					a_model = augment_action(step, np.copy(a_model))
					o, r, d_model, _  = env.step(a_model)
				else:
					break

			ep_lens_agent[i, j] = step + 1

			dist = np.sqrt(np.sum(np.square(o)))
			dist_to_center_agent[i, j] = dist

			# final_a_agent[i, j] = a_model

			for step in range(env.EPISODE_LENGTH_LIMIT):
				if not d_pi:
					a_pi = opt_env.pi_controller_action()
					a_pi = np.clip(a_pi, -1.0, 1.0)
					a_pi = augment_action(step, np.copy(a_pi))
					o_pi, r_pi, d_pi, _  = opt_env.step(a_pi)
				else:
					break

			ep_lens_pi[i, j] = step + 1

			dist = np.sqrt(np.sum(np.square(o_pi)))
			dist_to_center_pi[i, j] = dist

			# final_a_pi[i, j] = a_pi

	get_quartiles = lambda arr: df(arr).quantile([0.0, 0.25, 0.5, 0.75, 1.0], axis=1, interpolation='nearest').to_numpy()

	def plot_quartiles(xdata, agent_data, pi_data, title_name, xlabel, ylabel):
		fig, ax = plt.subplots(figsize=(10, 7))
		fig.suptitle(title_name, size=18)

		ax.plot(xdata, agent_data[2], c='b')
		ax.plot(xdata, agent_data[0], c='b', ls=':')
		ax.plot(xdata, agent_data[4], c='b', ls=':')
		ax.fill_between(xdata, agent_data[1], agent_data[3], color='b', alpha=0.4, hatch='\\')

		ax.plot(xdata, pi_data[2], c='r')
		ax.plot(xdata, pi_data[0], c='r', ls=':')
		ax.plot(xdata, pi_data[4], c='r', ls=':')
		ax.fill_between(xdata, pi_data[1], pi_data[3], color='r', alpha=0.4, hatch='\\')

		legend_handles = [Line2D([0], [0], c='b'),
						Line2D([0], [0], c='r'),
						Patch(facecolor='gray', alpha=0.4, hatch='\\'),
						Line2D([0], [0], c='gray', ls=':')]
		legend_labels = ['Agent Median', 'PI controller Median', '1st - 3rd quartile', 'Lower - Upper bounds']
		ax.legend(legend_handles, legend_labels, loc='upper left', prop={'size':15})
		ax.set_ylabel(ylabel, size=15)
		ax.set_xlabel(xlabel, size=15)
		ax.grid()
		fig.tight_layout()

		return ax, legend_handles, legend_labels

	get_savepath = lambda hajta: os.path.join('results', 'episode-statistics', f'{rl_type.lower()}_{hajta}_vs_nb-failures_{NB_EPS}-episodes.png')
	# def get_savepath()
	'''Episode length stats'''
	ela = get_quartiles(ep_lens_agent)
	elp = get_quartiles(ep_lens_pi)

	for table_nb_failures in (0, 1, 3, 5):
		if table_nb_failures in range_failures:
			idx = np.searchsorted(range_failures, table_nb_failures)
			print(f'{rl_type}')
			print(f'-> Number of failures = {range_failures[idx]}')
			print(f'\t-> Agent Mean ep.len. = {np.mean(ep_lens_agent[idx]):.4f}')
			print(f'\t-> Agent Std  ep.len. = {np.std(ep_lens_agent[idx]):.4f}')
			print(f'\t-> Agent Mean dtc = {np.mean(dist_to_center_agent[idx]):.4f}')
			print(f'\t-> Agent Std  dtc = {np.std(dist_to_center_agent[idx]):.4f}')
			print(f'\t-> PI Mean ep.len. = {np.mean(ep_lens_pi[idx]):.4f}')
			print(f'\t-> PI Std  ep.len. = {np.std(ep_lens_pi[idx]):.4f}')
			print(f'\t-> PI Mean dtc = {np.mean(dist_to_center_pi[idx]):.4f}')
			print(f'\t-> PI Std  dtc = {np.std(dist_to_center_pi[idx]):.4f}')

	# action_noise_sigmas *= 100.0

	title_name = f'{rl_type} Best policy evaluation\nEpisode length vs. Nb. actuator failures\nNb. episodes = {NB_EPS}'
	xlabel = 'Number of actuator failures'
	ylabel = 'Episode length'

	ax, *_ = plot_quartiles(range_failures, ela, elp, title_name, xlabel, ylabel)
	ax.set_ylim((-3, env.EPISODE_LENGTH_LIMIT + 3))

	plt.savefig(get_savepath('ep-len'))


	'''Distance to center (optimal point) stats'''
	dtca = get_quartiles(dist_to_center_agent)
	dtcp = get_quartiles(dist_to_center_pi)

	title_name = f'{rl_type} Best policy evaluation\nFinal distance to optimal point vs. Nb. actuator failures\nNb. episodes = {NB_EPS}'
	ylabel = '$\sqrt{\Delta Q_H^2 + \Delta Q_V^2}$'

	ax, handles, labels = plot_quartiles(range_failures, dtca, dtcp, title_name, xlabel, ylabel)
	ax.get_legend().remove()

	goal_line = np.sqrt(np.sum(np.square([env.Q_goal, env.Q_goal])))
	ax.axhline(goal_line, color='g', ls='--')

	handles.append(Line2D([0], [0], color='g', ls='--'))
	labels.append('Goal threshold')
	ax.legend(handles, labels, prop=dict(size=15), loc='best')

	plt.savefig(get_savepath('dtc'))

	# '''Action values at episode end'''
	# faa = get_quartiles(final_a_agent.reshape((NB_INTERVALS, -1)))
	# fap = get_quartiles(final_a_pi.reshape((NB_INTERVALS, -1)))
	# title_name = f'{rl_type} Best policy evaluation\nAction values at episode end vs. Action noise\nNb. episodes = {NB_EPS}'
	# ylabel = 'Action values (normalised)'
	# ax, *_ = plot_quartiles(action_noise_sigmas, faa, fap, title_name, xlabel, ylabel)
	# # ax.set_yscale('symlog', linthresh=0.1, linscale=2)
	# ax.grid(which='major', axis='y', color='gray')
	# ax.minorticks_on()
	# ax.grid(which='minor', axis='y', color='gray', linestyle='--', alpha=0.5)
	#
	# plt.savefig(get_savepath('act-vals'))


def run_test_on_best_agent(ALGO, TEST):
	# ALGO = ppo, naf2, td3, sac, sac-tfl, ae-dyna
	# TEST = action-noise, state-perturb, actfail
	env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
	                  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
	                  noise_std=0.0)

	if 'sac-tfl' in ALGO or 'ae-dyna' in ALGO:
		'''SAC-TFL/AE-DYNA'''
		import sys
		sys.path.append('D:\Code\FERMI_RL_Paper')
		from SAC_TFlayers import SAC as SACTFL

		model_type = SACTFL

		if 'sac-tfl' in ALGO:
			'''1. SAC-TFL'''
			model_par_dir = os.path.join('D:\Code\FERMI_RL_Paper', 'models')
			model_names = ['SAC-TFL_QFBNL_071421_014328']  # ,
			# 'SAC-TFL_QFBNL_071421_111615',
			# 'SAC-TFL_QFBNL_071421_014314',
			# 'SAC-TFL_QFBNL_071421_013557',
			# 'SAC-TFL_QFBNL_071321_223236']
			best_model_steps = [81500]  # , 46800, 46200, 33000, 47900]
			name_parser = lambda model_path, step: f'{os.path.split(model_path)[-1]}_{step}_steps'

		elif 'ae-dyna' in ALGO:
			'''2. AE-DYNA-SAC'''
			model_par_dir = 'D:\\Code\\FERMI_RL_Paper\\models_aedyna'
			# best_model_steps = [82600, 155120]
			best_model_steps = [155120]
			# model_names = ['AE-DYNA_QFBNL_071921_035442', 'AE-DYNA_QFBNL_072021_024119']
			model_names = ['AE-DYNA_QFBNL_072021_024119']

			def name_parser(model_path, step):
				nonlocal model_par_dir
				search_dir = os.path.join(model_par_dir, model_path, 'agent')
				for item in os.listdir(search_dir):
					if str(step) in item:
						return os.path.join('agent', item)
	elif 'sac' in ALGO:
		'''SAC'''
		model_par_dir = 'models'#'models_sac_hparams'
		model_type = SAC
		model_names = ['SAC_QFBNL_060821_2055']  # ,
		#                'SAC_QFBNL_060921_0627',
		#                'SAC_QFBNL_060921_0928',
		#                'SAC_QFBNL_060921_0049',
		#                'SAC_QFBNL_060921_0011',
		#                'SAC_QFBNL_060921_0909']
		# best_model_steps = [75000, 30000, 67000, 72000, 65000, 39000]
		best_model_steps = [75000]
		name_parser = lambda model_path, step: f'{os.path.split(model_path)[-1]}_{step}_steps.zip'
	elif 'ppo' in ALGO:
		'''PPO'''
		from stable_baselines import PPO2
		model_par_dir = 'models'#'models_thesis'
		model_type = PPO2
		# model_names = [f'PPO_QFBNL_050721_{item}' for item in ('0948', '0059', '1040')]#('0059', '0922', '0948', '1014', '1040')]
		model_names = [f'PPO_QFBNL_050721_0948']
		best_model_steps = [83000]  # , 73000, 60000]
		name_parser = lambda model_path, step: f'{os.path.split(model_path)[-1]}_{step}_steps.zip'
	elif 'td3' in ALGO:
		'''TD3'''
		model_par_dir = 'models'#'models_td3_hparams'
		model_type = TD3

		model_names = [
			'TD3_QFBNL_061021_0054']  # , 'TD3_QFBNL_060921_1825', 'TD3_QFBNL_060921_2244', 'TD3_QFBNL_060921_2227', 'TD3_QFBNL_060921_1204']
		best_model_steps = [15000]  # , 7000, 23000, 15000, 59000]
		#
		# best_model_steps = np.arange(70000, 80001, 1000)
		# model_names = ['TD3_QFBNL_060921_1204'] * len(best_model_steps)
		name_parser = lambda model_path, step: f'{os.path.split(model_path)[-1]}_{step}_steps.zip'
	elif 'naf2' in ALGO:
		'''NAF2 - requires TF2'''
		from NAF2.naf2 import NAF2
		model_par_dir = 'models'#'models_thesis'
		model_type = NAF2
		model_names = ['NAF2_QFBNL_051321_2157']  # ,
		# 'NAF2_QFBNL_051421_0722',
		# 'NAF2_QFBNL_051421_2113',
		# 'NAF2_QFBNL_051521_0901',
		# 'NAF2_QFBNL_051721_1251']
		best_model_steps = [88000]  # , 115000, 98000, 99000, 96000]
		name_parser = lambda model_name, step: f"step_{f'{step}'.zfill(4)}"

	rl_type = ALGO.upper()
	if 'action-noise' in TEST:
		env_kwargs['perturb_state'] = False
		env = QFBNLEnv(**env_kwargs)
		opt_env = QFBNLEnv(**env_kwargs)

		for action_noise_sigma in [0.0 , 0.1, 0.25, 0.5]:
			for model_name, best_model_step in zip(model_names, best_model_steps):
				if action_noise_sigma == 0.0:
					title_name = f"{rl_type} Policy Evaluation - No state/action noise"
				else:
					title_name = f"{rl_type} Policy Evaluation - {action_noise_sigma * 100.0:.1f}% action Gaussian noise"
				save_episode_trace_only_best_step_action_noise(os.path.join(model_par_dir, model_name),
				                                               best_model_step, name_parser, model_type,
				                                               env, opt_env,
				                                               title_name, action_noise_sigma,
				                                               save_dir=os.path.join('prab_results', 'ps_images'))#,
				                                                                     # 'action_noise',
				                                                                     # rl_type))
	elif 'state-perturb' in TEST:
		action_noise_sigma = 0.0
		env_kwargs['perturb_state'] = True
		env = QFBNLEnv(**env_kwargs)
		opt_env = QFBNLEnv(**env_kwargs)

		for model_name, best_model_step in zip(model_names, best_model_steps):
			title_name = f"{rl_type} Best Policy Evaluation - State perturbed by 50 Hz harmonics"
			save_episode_trace_only_best_state_perturb(os.path.join(model_par_dir, model_name),
			                                           best_model_step, name_parser, model_type,
			                                           env, opt_env,
			                                           title_name, action_noise_sigma,
			                                           save_dir=os.path.join('prab_results'))#,
			                                                                     # 'perturb_state',
			                                                                     # rl_type))
	elif 'actfail' in TEST:
		action_noise_sigma = 0.0
		env_kwargs['perturb_state'] = False
		env = QFBNLEnv(**env_kwargs)
		env2 = QFBNLEnv(**env_kwargs)
		opt_env = QFBNLEnv(**env_kwargs)
		opt_env2 = QFBNLEnv(**env_kwargs)

		for model_name, best_model_step in zip(model_names, best_model_steps):
			title_name = f"{rl_type} Best Policy Evaluation - Actuator failures"
			save_episode_trace_only_best_act_failure(os.path.join(model_par_dir, model_name),
			                                         best_model_step, name_parser, model_type,
			                                         env, opt_env, env2, opt_env2,
			                                         title_name, action_noise_sigma,
			                                         save_dir=os.path.join('prab_results')),
			                                                               # 'act_failure',
			                                                               # rl_type))

	elif 'plot-episode-statistics' in TEST:
		action_noise_sigma = 0.0

		env_kwargs['perturb_state'] = False
		env = QFBNLEnv(**env_kwargs)
		opt_env = QFBNLEnv(**env_kwargs)
		for model_name, best_model_step in zip(model_names, best_model_steps):
			plot_episode_statistics_best_step_action_noise(os.path.join(model_par_dir, model_name),
														   best_model_step, name_parser, model_type,
														   env, opt_env)
	elif 'save-episode-statistics' in TEST:
		action_noise_sigma = 0.0

		env_kwargs['perturb_state'] = False
		env = QFBNLEnv(**env_kwargs)
		opt_env = QFBNLEnv(**env_kwargs)
		for model_name, best_model_step in zip(model_names, best_model_steps):
			data = calculate_episode_statistics_best_step_action_noise(os.path.join(model_par_dir, model_name),
																	   best_model_step, name_parser, model_type,
																	   env, opt_env)
			NB_INTERVALS, NB_EPS = data['ep_lens_agent'].shape
			with open(f'{ALGO}_{TEST}_{NB_INTERVALS}-intervals_{NB_EPS}-episodes.pkl', 'wb') as f:
				pkl.dump(data, f)

	elif 'plot-episode-statistics-act-fail' in TEST:
		action_noise_sigma = 0.0

		env_kwargs['perturb_state'] = False
		env = QFBNLEnv(**env_kwargs)
		opt_env = QFBNLEnv(**env_kwargs)
		for model_name, best_model_step in zip(model_names, best_model_steps):
			get_episode_statistics_best_step_actuator_failure(os.path.join(model_par_dir, model_name),
															  best_model_step, name_parser, model_type,
															  env, opt_env)



def get_results():
	""""""
	""" tf1x """
	# for algo in ('ppo',):# 'sac', 'td3'):
	# 	# for test in ('action-noise', 'state-perturb', 'actfail'):
	# 	for test in ('save-episode-statistics',):
	# 		run_test_on_best_agent(algo, test)

	""" tf2x """
	for algo in ('naf2',):#'ae-dyna', 'sac-tfl',
	# for algo in ('naf2',):
	# # 	for test in ( 'actfail', 'action-noise', 'state-perturb'):
		for test in ('save-episode-statistics',):
			run_test_on_best_agent(algo, test)


def main2():
	# np.random.seed(456)
	# tf.random.set_seed(456)
	# random.seed(456)

	env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
	                  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
	                  perturb_state=False,
	                  noise_std=0.0)
	env = QFBNLEnv(**env_kwargs)

	'''SACTFL'''
	import sys
	sys.path.append('D:\Code\FERMI_RL_Paper')
	from SAC_TFlayers import SAC as SACTFL

	model_type = SACTFL

	# par_dir = os.path.join('D:\Code\FERMI_RL_Paper', 'models')
	# model_names = ['SAC-TFL_QFBNL_071321_223236',
	#                'SAC-TFL_QFBNL_071421_013557',
	#                'SAC-TFL_QFBNL_071421_014314',
	#                'SAC-TFL_QFBNL_071421_014328',
	#                'SAC-TFL_QFBNL_071421_111615']

	par_dir = 'D:\\Code\\FERMI_RL_Paper\\models_aedyna'
	model_names = [item for item in os.listdir(par_dir) if 'AE-DYNA' in item]

	'''SAC'''
	# par_dir = 'models_sac_hparams'
	# model_type = SAC
	# # model_names = [f'SAC_QFBNL_060921_{timestr}' for timestr in ['0815', '0049', '0011', '0909', '0928']]
	# model_names = [f'SAC_QFBNL_060921_{timestr}' for timestr in ['0741', '0627', '0143']]
	# model_names.extend([f'SAC_QFBNL_060821_{timestr}' for timestr in ['2055', '2244', '2131']])
	# for model_name in ['SAC_QFBNL_060921_0815', 'SAC_QFBNL_060921_0049', 'SAC_QFBNL_060921_0011', 'SAC_QFBNL_060921_0909', 'SAC_QFBNL_060921_0928']:
	# for model_name in ['TD3_QFBNL_060921_2244', 'TD3_QFBNL_061021_0054', 'TD3_QFBNL_060921_2227', 'TD3_QFBNL_060921_1204', 'TD3_QFBNL_060921_1825']:

	'''TD3'''
	# par_dir = 'models_td3_hparams'
	# model_type = TD3
	# # model_names = [f'TD3_QFBNL_06{item}' for item in ('0921_2244', '1021_0054', '0921_2227', '0921_1204', '0921_1825')]
	# model_names = [f'TD3_QFBNL_060921_1825']

	for model_name in model_names:
		models_dir = os.path.join(par_dir, model_name, 'agent')
		get_best_models(models_dir=models_dir, model_type=model_type, env=env, txt_filename_suffix='')


def test_with_simulated_50hz_state_perturbation():
	seed = 123
	np.random.seed(seed)
	if '1' in tf.__version__.split('.'):
		tf.random.set_random_seed(seed)
	else:
		tf.random.set_seed(seed)
	random.seed(seed)
	# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
	                  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
	                  perturb_state=True,
	                  noise_std=0.0)
	env = QFBNLEnv(**env_kwargs)
	opt_env = QFBNLEnv(**env_kwargs)

	'''TF1 agents'''
	sac_model_names = ['SAC_QFBNL_060821_2055']
	sac_model_par_dirs = ['models_sac_hparams'] * len(sac_model_names)
	sac_best_model_steps = [75000]
	sac_model_types = [SAC] * len(sac_model_names)
	sac_name_parser = lambda model_name, step: f'{model_name}_{step}_steps.zip'
	sac_name_parsers = [sac_name_parser] * len(sac_model_names)

	td3_model_names = ['TD3_QFBNL_061021_0054', 'TD3_QFBNL_060921_1825', 'TD3_QFBNL_060921_2244',
	                   'TD3_QFBNL_060921_2227', 'TD3_QFBNL_060921_1204']
	td3_model_par_dirs = ['models_td3_hparams'] * len(td3_model_names)
	td3_best_model_steps = [15000, 7000, 23000, 15000, 59000]
	td3_model_types = [TD3] * len(td3_model_names)
	td3_name_parsers = [sac_name_parser] * len(td3_model_names)

	ppo_model_names = [f'PPO_QFBNL_050721_{item}' for item in ('0948', '0059', '1040')]
	ppo_model_par_dirs = ['models_thesis'] * len(ppo_model_names)
	ppo_best_model_steps = [83000, 73000, 60000]
	ppo_model_types = [PPO2] * len(ppo_model_names)
	ppo_name_parsers = [sac_name_parser] * len(ppo_model_names)

	model_paths = [os.path.join(path, name) for path, name in
	               zip([*sac_model_par_dirs, *td3_model_par_dirs, *ppo_model_par_dirs],
	                   [*sac_model_names, *td3_model_names, *ppo_model_names])]
	best_model_steps = [*sac_best_model_steps, *td3_best_model_steps, *ppo_best_model_steps]
	model_types = [*sac_model_types, *td3_model_types, *ppo_model_types]
	name_parsers = [*sac_name_parsers, *td3_name_parsers, *ppo_name_parsers]

	'''SAC-TFL'''
	# import sys
	# sys.path.append('D:\Code\FERMI_RL_Paper')
	# from SAC_TFlayers import SAC as SACTFL
	#
	# model_type = SACTFL
	#
	# model_par_dir = os.path.join('D:\Code\FERMI_RL_Paper', 'models')
	# model_names = ['SAC-TFL_QFBNL_071421_014328',
	#                'SAC-TFL_QFBNL_071421_111615',
	#                'SAC-TFL_QFBNL_071421_014314',
	#                'SAC-TFL_QFBNL_071421_013557',
	#                'SAC-TFL_QFBNL_071321_223236']
	# best_model_steps = [81500, 46800, 46200, 33000, 47900]
	# name_parser = lambda model_name, step: f'{model_name}_{step}_steps'

	# model_par_dir = 'D:\\Code\\FERMI_RL_Paper\\models_aedyna'
	# best_model_steps = [155120]#82600]
	# model_names = ['AE-DYNA_QFBNL_072021_024119']
	#
	# def name_parser(model_path, step):
	#     search_dir = os.path.join(model_path, 'agent')
	#     for item in os.listdir(search_dir):
	#         if str(step) in item:
	#             return os.path.join('agent', item)

	'''NAF2 - requires TF2'''
	# from NAF2.naf2 import NAF2
	# model_par_dir = 'models_thesis'
	# model_type = NAF2
	# model_names = ['NAF2_QFBNL_051321_2157',
	#                'NAF2_QFBNL_051421_0722',
	#                'NAF2_QFBNL_051421_2113',
	#                'NAF2_QFBNL_051521_0901',
	#                'NAF2_QFBNL_051721_1251']
	# best_model_steps = [88000, 115000, 98000, 99000, 96000]
	# name_parser = lambda model_name, step: f"step_{f'{step}'.zfill(4)}"

	for action_noise_sigma in [0.0, 0.1, 0.25, 0.5]:
		test_state_perturbation(model_paths, best_model_steps, name_parsers, model_types,
		                        env, opt_env, action_noise_sigma)


def dilemma_of_naf2_last_action():
	from NAF2.naf2 import NAF2
	model_par_dir = 'models_thesis'
	model_type = NAF2
	# model_names = ['NAF2_QFBNL_051321_2157',
	#                'NAF2_QFBNL_051421_0722',
	#                'NAF2_QFBNL_051421_2113',
	#                'NAF2_QFBNL_051521_0901',
	#                'NAF2_QFBNL_051721_1251']
	model_names = ['NAF2_QFBNL_051721_1251']
	# best_model_steps = [88000, 115000, 98000, 99000, 96000]
	best_model_steps = [96000]
	name_parser = lambda model_name, step: f"step_{f'{step}'.zfill(4)}"

	env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
	                  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
	                  perturb_state=False,
	                  noise_std=0.0)
	env = QFBNLEnv(**env_kwargs)
	opt_env = QFBNLEnv(**env_kwargs)

	n_act = opt_env.act_dimension
	n_obs = opt_env.obs_dimension

	for action_noise_sigma in [0.0]:  # , 0.1, 0.25, 0.5]:
		for model_name, best_model_step in zip(model_names, best_model_steps):
			noise = lambda: np.random.normal(0.0, action_noise_sigma, env.act_dimension)
			rl_type = model_name.split('_')[0]

			model_path = os.path.join(model_par_dir, model_name)
			model = model_type.load(os.path.join(model_path, name_parser(model_name, best_model_step)))

			if action_noise_sigma == 0.0:
				title_name = f"{rl_type} Policy Evaluation - No state/action noise"
			else:
				title_name = f"{rl_type} Policy Evaluation - {action_noise_sigma * 100.0:.1f}% action Gaussian noise"

			o = env.reset()
			opt_env.reset(o.copy())
			opt_env.pi_controller_warm_up()
			d_model, d_pi = False, False
			rewards_model, rewards_pi = [], []
			actions_model = np.zeros(shape=(0, n_act))
			actions_pi = np.zeros(shape=(0, n_act))
			states_model = np.zeros(shape=(0, n_obs))
			states_pi = np.zeros(shape=(0, n_obs))

			FREEZE_ACTION_FOR_STEPS = 100

			for step in range(env.EPISODE_LENGTH_LIMIT):
				if not d_model:
					a_model = model.predict(o.copy(), deterministic=True)[0]  # + noise()
					if 'SAC-TFL' in rl_type or 'AE-DYNA' in rl_type:
						a_model = model.predict(o.copy(), deterministic=True)  # + noise()  # only for SACTFL
					a_model = np.clip(a_model, -1.0, 1.0)
					o, r, d_model, _ = env.step(a_model)
					actions_model = np.vstack([actions_model, a_model])
					states_model = np.vstack([states_model, o.copy()])
					rewards_model.append(r)

				if not d_pi:
					a_pi = opt_env.pi_controller_action() + noise()
					a_pi = np.clip(a_pi, -1.0, 1.0)
					o_pi, r_pi, d_pi, _ = opt_env.step(a_pi)
					actions_pi = np.vstack([actions_pi, a_pi])
					states_pi = np.vstack([states_pi, o_pi.copy()])
					rewards_pi.append(r_pi)

			for _ in range(FREEZE_ACTION_FOR_STEPS):
				last_action = actions_model[-1]
				o, r, d_model, _ = env.step(last_action)

				actions_model = np.vstack([actions_model, last_action])
				states_model = np.vstack([states_model, o.copy()])
				rewards_model.append(r)

			fig, (ax, ax2) = plt.subplots(2)
			# ax.plot(states_model.T[0], states_model.T[1], marker='x', c='b')
			ax.plot(np.arange(states_model.shape[0]), states_model.T[0], marker='x', c='b')
			ax.plot(np.arange(states_model.shape[0]), states_model.T[1], marker='x', c='b')
			ax.set_xlabel('Delta Tune Horizontal')
			ax.set_ylabel('Delta Tune Vertical')
			ax.axhline(0.0, color='k', ls='--')
			ax.axvline(states_model.shape[0] - FREEZE_ACTION_FOR_STEPS, color='c')

			ax2.plot(np.arange(actions_model.shape[0]), np.array(actions_model), c='r')
			ax2.set_xlabel('Steps')
			ax2.set_ylabel('Normalised Actions')
			ax2.axhline(0.0, color='k', ls='--')
			plt.show(block=False)
			fig.tight_layout()


if __name__ == '__main__':
	# get_results()
	frontiers_merge_high_stats_episode_statistics()