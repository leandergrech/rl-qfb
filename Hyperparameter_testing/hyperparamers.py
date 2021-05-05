import os
import numpy as np
import csv
import matplotlib.pyplot as plt

stats_file = 'run-SAC_QFBNL_050121_2112_1-tag-episode_length.csv'
with open(stats_file, 'r', newline='\n') as csvfile:
    file = csv.reader(csvfile, delimiter=',')
    steps = []
    ep_lens = []
    for i, row in enumerate(file):
        if i == 0:
            continue
        print(row)
        steps.append(int(row[1]))
        ep_lens.append(float(row[2]))

alpha = 0.9
ep_lens_smooth=[ep_lens[0]]
for i, ep_len in enumerate(ep_lens[1:]):
    ep_lens_smooth.append(ep_lens_smooth[-1] * (alpha) + ep_len * (1 - alpha))

fig, ax1 = plt.subplots()
ax1.plot(steps, ep_lens, label='ep_lens', color='c')
ax1.plot(steps, ep_lens_smooth, label='ep_lens_smooth', color='g')
ax2 = ax1.twinx()
ax2.plot((steps + np.mean(np.diff(steps))/2)[:-1], np.diff(ep_lens), label='ep_lens diff', color='red')
ax2.plot((steps + np.mean(np.diff(steps))/2)[:-1], np.diff(ep_lens_smooth), label='ep_lens_smooth diff', color='orange')
fig.legend(loc='lower left')
plt.tight_layout()
plt.show()


