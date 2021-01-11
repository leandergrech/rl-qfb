import pickle as pkl
import matplotlib.pyplot as plt

with open('buffer_storage_tmp.pkl', 'rb') as f:
    data = pkl.load(f)

# print(list(data.keys()))
print(data['done'])
print(data['episode_snips'])