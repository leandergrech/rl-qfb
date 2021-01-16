import os

import tensorflow as tf
from stable_baselines import SAC

model_name = 'SAC_QFB_011321_1618_45000_steps.zip'
model = SAC.load(os.path.join('models', model_name))
pi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/pi')

print(pi)


