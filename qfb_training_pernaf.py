from pernaf.naf import NAF
import tensorflow as tf

from envs.qfb_env import QFBEnv

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':
	env = QFBEnv()
	sess = tf.Session()
	naf = NAF(sess, env, )