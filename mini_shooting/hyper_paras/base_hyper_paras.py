"""
This is base hyper-parameters, you can modify them here.
"""


class BaseHyperparameters(object):
    def __init__(self):
        # self.N_FEATURES = [210, 160, 3]  # Without cropping and stack.
        # self.N_FEATURES = [4, 80, 80]  # With cropping and stack.
        self.N_ACTIONS = 13
        self.IMAGE_SIZE = 9
        self.N_STACK = 4
        self.model = 'null'

        self.MAX_EPISODES = 50001  # 50001 : 500
        self.LEARNING_RATE = 0.00001
        self.INITIAL_EXPLOR = 0  # 0 : 0.5
        self.FINAL_EXPLOR = 0.9
        self.FINAL_EXPLOR_FRAME = 45000  # 45000 : 450
        self.DISCOUNT_FACTOR = 0.99
        self.MINIBATCH_SIZE = 32  # 32 : 8
        self.REPLY_START_SIZE = 1000  # 400 : 100
        self.REPLY_MEMORY_SIZE = 200000
        self.TARGET_NETWORK_UPDATE_FREQUENCY = 50000  # 50000 : 150

        # log and output
        self.WEIGHTS_SAVER_ITER = 4000  # 4000 : 200
        self.OUTPUT_SAVER_ITER = 2000  # 2000 : 100
        self.OUTPUT_GRAPH = False
        self.SAVED_NETWORK_PATH = './logs/network/'
        self.LOGS_DATA_PATH = './logs/data/'
        self.SAVED_NETWORK_PATH_BACK = './backup/network/'
        self.LOGS_DATA_PATH_BACK = './backup/data/'

        # Class Memory in pri_dqn
        self.M_NONZERO_EPSILON = 0.01  # small amount to avoid zero priority
        self.M_ALPHA = 0.6  # [0~1] convert the importance of TD error to priority
        self.M_INITIAL_BETA = 0.4  # importance-sampling, from initial value increasing to 1
        self.M_FINAL_BETA = 1
        self.M_FINAL_BETA_FRAME = 45000  # 45000 : 450
        self.M_ABS_ERROR_UPPER = 1.  # clipped abs error
