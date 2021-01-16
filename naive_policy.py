from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

from envs.qfb_env import QFBEnv

if __name__ == '__main__':
    env = QFBEnv()

    n_obs = env.obs_dimension
    n_act = env.act_dimension

    model = Sequential()
    model.add(Input(shape=(n_obs)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(n_act, activation=))

