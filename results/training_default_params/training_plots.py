import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import csv
from collections import defaultdict


def spit_plot_diff_random_seeds(algo, param, ylabel, par_dir=None):
    if par_dir is None:
        custom_par_dir = False
        par_dir = algo
    else:
        custom_par_dir = True

    # Get ep_len, ep_ret & succ sub-directories
    sub_dirs = os.listdir(par_dir)

    desired_steps = np.arange(0, 100001, 100)

    # Attach full paths to data
    # file_paths = defaultdict(list)
    file_paths = []
    # for sub_dir in sub_dirs:
    #     if not'.zip' in sub_dir:
    for file in os.listdir(os.path.join(par_dir, param)):
        # file_paths[sub_dir].append(os.path.join(par_dir, sub_dir, file))
        file_paths.append(os.path.join(par_dir, param, file))

    # Read the CSV files
    val_list = []
    step_list = []
    for path in file_paths:
        val_list.append([])
        step_list.append([])
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)     # Consume column names
            for i, row in enumerate(csvreader):
                step_list[-1].append(int(row[1]))
                val_list[-1].append(float(row[-1]))

    # Transform to desired x-axis
    # FYI: step_list and val_list can be used with the same index
    temp = np.zeros(shape=(len(val_list), len(desired_steps)))
    for i, desstep in enumerate(desired_steps):
        for j, steps in enumerate(step_list):
            val_idx = np.argmin(np.abs(np.subtract(steps, desstep)))
            temp[j][i] = val_list[j][val_idx]

    val_list = temp

    min_len = np.inf
    for item in val_list:
        min_len = min(min_len, len(item))

    vals = np.zeros(shape=(0, min_len))
    for item in val_list:
        vals = np.vstack([vals, np.array(item[:min_len], dtype=np.float)])

    stat_type_label = 'Median'
    param_mea = np.median(vals, axis=0)

    param_min = np.min(vals, axis=0)
    param_max = np.max(vals, axis=0)

    fig, ax = plt.subplots()
    x = desired_steps
    ax.plot(x, param_mea, label=stat_type_label)
    ax.fill_between(x, param_min, param_max, alpha=0.5, label='Min-Max')

    if param == 'ep_ret':
        ax.set_yscale('symlog')

        maxmedret = max(param_mea)
        step_maxmedret = x[np.argmax(param_mea)]
        ax.axhline(maxmedret, color='tab:orange', label=f'Max({stat_type_label})')
        ax.axvline(step_maxmedret, color='tab:orange', lw=0.2)

        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, maxmedret, f"{maxmedret:.1f}", color="tab:orange", transform=trans,
                ha="right", va="center")
        trans = transforms.blended_transform_factory(
            ax.transData, ax.get_xticklabels()[0].get_transform())
        ax.text(step_maxmedret, 0.05, f"{int(np.round(step_maxmedret))}", color="tab:orange", transform=trans,
                ha="center", va="center")

    ax.grid(axis='both', which='both', alpha=0.3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Training Steps')
    ax.set_title(algo)
    ax.legend(loc='best')

    fig.tight_layout()

    # plt.show()
    if custom_par_dir:
        save_path = os.path.join(par_dir, f'{algo}_{param}.pdf')
    else:
        save_path = f'{algo}_{param}.pdf'
    plt.savefig(save_path)
    print(f'Saved to: {save_path}')

    # fig, ax = plt.subplots()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface()

def spit_plot_diff_best_hparams(algo, param, ylabel, index_to_agent_hint, par_dir=None):
    if par_dir is None:
        custom_par_dir = False
        par_dir = algo
    else:
        custom_par_dir = True

    desired_steps = np.arange(0, 80001, 1000)

    # Attach full paths to data
    file_paths = []
    for file in os.listdir(os.path.join(par_dir, param)):
        # file_paths[sub_dir].append(os.path.join(par_dir, sub_dir, file))
        file_paths.append(os.path.join(par_dir, param, file))

    # Read the CSV files
    val_list = []
    step_list = []
    for path in file_paths:
        val_list.append([])
        step_list.append([])
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)     # Consume column names
            for i, row in enumerate(csvreader):
                step = int(row[1])
                val = float(row[-1])

                if step % 1000 != 0:
                    continue

                step_list[-1].append(step)
                val_list[-1].append(val)

    # Transform to desired x-axis
    # FYI: step_list and val_list can be used with the same index
    temp = np.zeros(shape=(len(val_list), len(desired_steps)))
    for i, desstep in enumerate(desired_steps):
        for j, steps in enumerate(step_list):
            val_idx = np.argmin(np.abs(np.subtract(steps, desstep)))
            temp[j][i] = val_list[j][val_idx]

    vals = temp

    fig, ax = plt.subplots()
    x = desired_steps

    for i in range(len(index_to_agent_hint)):
        for val, fp in zip(vals, file_paths):
            if index_to_agent_hint[i] in fp:
                ax.plot(x, val, label=f'{algo}#{i}', lw=1, alpha=0.8)

    # for val, fp in zip(vals, file_paths):
    #     for k, v in index_to_agent_hint.items():
    #         if k in fp:
    #             ax.plot(x, val, label=f'{algo}#{v}')

    if param == 'ep_ret':
        ax.set_yscale('symlog')
        c = 'lime'

        maxmedret = np.max(vals.flatten())
        step_maxmedret = x[np.argmax(vals.flatten()) % vals.shape[1]]
        ax.axhline(maxmedret, color=c, label=f'Maximum')
        ax.axvline(step_maxmedret, color=c, lw=0.2)

        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, maxmedret-0.3, f"{maxmedret:.1f}", color=c, transform=trans,
                ha="right", va="center")
        trans = transforms.blended_transform_factory(
            ax.transData, ax.get_xticklabels()[0].get_transform())
        ax.text(step_maxmedret, 0.05, f"{int(np.round(step_maxmedret))}", color=c, transform=trans,
                ha="center", va="center")
    ax.grid(axis='both', which='both', alpha=0.3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Training Steps')
    ax.set_title(algo)
    ax.legend(loc='best')

    fig.tight_layout()

    if custom_par_dir:
        save_path = os.path.join(par_dir, f'{algo}_{param}.pdf')
    else:
        save_path = f'{algo}_{param}.pdf'
    plt.savefig(save_path)
    print(f'Saved to: {save_path}')


def plot_all_agents():
    par_dir = 'D:\Code\FERMI_RL_Paper\models'
    # algos = ['NAF2', 'PPO', 'TD3', 'SAC']
    algos = ['SAC-TFL']
    # algos = ['PPO']

    params = ['ep_len', 'ep_ret', 'succ']
    ylabels = ['Episode Length', 'Episode Return ($\gamma\,=\,1$)', 'Success Rate (%)']

    for algo in algos:
        for param, ylabel in zip(params, ylabels):
            spit_plot_diff_random_seeds(algo, param, ylabel)

def plot_all_hparam_agents():
    for param, ylabel in zip(('ep_len', 'ep_ret', 'succ'),
                             ('Episode Length', 'Episode Return ($\gamma\,=\,1$)', 'Success Rate (%)')):
        # spit_plot_diff_best_hparams(algo='SAC',
        #                             param=param,
        #                             ylabel=ylabel,
        #                             hint_to_agent_index={'0815':0, '0049':1, '0011':2, '0909':3, '0928':4},
        #                             par_dir='SAC_best_hparam')
        spit_plot_diff_best_hparams(algo='TD3',
                                    param=param,
                                    ylabel=ylabel,
                                    index_to_agent_hint={0:'0054', 1:'1825', 2:'2244', 3:'2227', 4:'1204'},
                                    par_dir='TD3_best_hparam')

    # plt.show()

if __name__ == '__main__':
    plot_all_agents()