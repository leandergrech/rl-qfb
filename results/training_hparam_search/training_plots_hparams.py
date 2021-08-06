import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import csv
from collections import defaultdict

def hparam_comparison():
    algo = 'TD3'
    param_ep_len = 'ep_len'
    param_succ = 'succ'

    # Get ep_len values from all run*.csv
    par_dir = os.path.join(algo, param_ep_len)
    ep_len_list = []
    step_list = []
    agent_code_list = []
    for filename in os.listdir(par_dir):
        full_path = os.path.join(par_dir, filename)
        agent_code_list.append(filename.split('-')[1][:-2])
        ep_len_list.append([])
        step_list.append([])
        with open(full_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)  # Consume column names
            for i, row in enumerate(csvreader):
                step_list[-1].append(int(row[1]))
                ep_len_list[-1].append(float(row[-1]))

    # Find length of expected results
    min_len = np.inf
    for item in ep_len_list:
        min_len = min(min_len, len(item))
    ep_lens = np.zeros(shape=(0, min_len))
    for item in ep_len_list:
        ep_lens = np.vstack([ep_lens, np.array(item[:min_len], dtype=np.float)])

    # Find average of ep_len
    ave_of_ep_len = np.mean(ep_lens, axis=1)

    # Sort some lists wrt. to ave_of_ep_len
    # indices = np.arange(len(ave_of_ep_len))
    # sorted_indices = [x for _, x in sorted(zip(ave_of_ep_len, indices))]
    sorted_agent_codes_wrt_ep_len = [x for _, x in sorted(zip(ave_of_ep_len, agent_code_list))]
    sorted_ave_ep_len = sorted(ave_of_ep_len)

    # Get successful agents
    par_dir = os.path.join(algo, param_succ)
    successful_agents = []
    succ_list = []
    for filename in os.listdir(par_dir):
        full_path = os.path.join(par_dir, filename)
        succ_list.append([])
        with open(full_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)  # Consume column names
            for i, row in enumerate(csvreader):
                succ_list[-1].append(float(row[-1]))

        if max(succ_list[-1]) == 100.0:
            successful_agents.append(filename.split('-')[1][:-2])

    # Read hparams from algo/save_logs.txt
    lr_list = []
    bas_list = []
    bus_list = []
    tau_list = []
    for agent_code in sorted_agent_codes_wrt_ep_len:
        with open(os.path.join(algo, 'save_logs.txt'), 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                elif agent_code in line:
                    read_lr = False
                    read_bas = False
                    read_bus = False
                    read_tau = False
                    while (not read_lr) or (not read_bus) or (not read_bas) or (not read_tau):
                        line = f.readline()
                        if 'learning_rate' in line:
                            lr_list.append(float(line.split('=')[-1]))
                            read_lr = True
                        elif 'buffer_size' in line:
                            bus_list.append(int(line.split('=')[-1]))
                            read_bus = True
                        elif 'batch_size' in line:
                            bas_list.append(int(line.split('=')[-1]))
                            read_bas = True
                        elif 'tau' in line:
                            tau_list.append(float(line.split('=')[-1]))
                            read_tau = True
                    break

    # Write sorted info in algo/save_logs_sorted_wrt_ep_len.txt
    with open(os.path.join(algo, 'save_logs_successful_sorted_wrt_ep_len.txt'), 'w') as f:
        for i, (agent_code, lr, bas, bus, tau) in enumerate(zip(sorted_agent_codes_wrt_ep_len, lr_list, bas_list, bus_list, tau_list)):
            if agent_code not in successful_agents:
                continue

            f.write(f'-> {agent_code}\n')
            f.write(f' `-> learning_rate = {lr}\n')
            f.write(f' `-> batch_size = {bas}\n')
            f.write(f' `-> buffer_size = {bus}\n')
            f.write(f' `-> tau = {tau}\n')
            f.write(f' `-> ave_ep_len = {sorted_ave_ep_len[i]}\n\n')

def spit_plot_diff_random_seeds(algo, param, ylabel):
    sub_dirs = os.listdir(algo)

    file_paths = defaultdict(list)

    for sub_dir in sub_dirs:
        for file in os.listdir(os.path.join(algo, sub_dir)):
            file_paths[sub_dir].append(os.path.join(algo, sub_dir, file))

    val_list = []
    for path in file_paths[param]:
        val_list.append([])
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)     # Consume column names
            for i, row in enumerate(csvreader):
                val_list[-1].append(row[-1])

    min_len = np.inf
    for item in val_list:
        min_len = min(min_len, len(item))

    vals = np.zeros(shape=(0, min_len))
    for item in val_list:
        vals = np.vstack([vals, np.array(item[:min_len], dtype=np.float)])

    param_med = np.median(vals, axis=0)
    param_min = np.min(vals, axis=0)
    param_max = np.max(vals, axis=0)

    fig, ax = plt.subplots()
    x = np.linspace(0, int(1e5), min_len)
    ax.plot(x, param_med, label='Median')
    ax.fill_between(x, param_min, param_max, alpha=0.5, label='Min-Max')

    if param == 'ep_ret':
        ax.set_yscale('symlog')

        maxmedret = max(param_med)
        step_maxmedret = x[np.argmax(param_med)]
        ax.axhline(maxmedret, color='tab:orange', label='Max(Median)')
        ax.axvline(step_maxmedret, color='tab:orange', lw=0.1, ls='dashed')

        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, maxmedret, f"{maxmedret:.1f}", color="tab:orange", transform=trans,
                ha="right", va="center")

    ax.grid(axis='both', which='both')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Training Steps')
    ax.set_title(algo)
    ax.legend(loc='best')

    fig.tight_layout()

    plt.savefig(f'{algo}_{param}.pdf')

def default_params_plots():
    algos = ['NAF2', 'PPO', 'TD3']

    params = ['ep_len', 'ep_ret', 'succ']
    ylabels = ['Episode Length', 'Episode Return ($\gamma\,=\,1$)', 'Success Rate (%)']

    for algo in algos:
        for param, ylabel in zip(params, ylabels):
            spit_plot_diff_random_seeds(algo, param, ylabel)

if __name__ == '__main__':
    hparam_comparison()
