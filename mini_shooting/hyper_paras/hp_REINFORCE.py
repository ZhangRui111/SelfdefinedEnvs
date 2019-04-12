class Hyperparameters():
    def __init__(self):
        # self.N_FEATURES = [210, 160, 3]  # Without cropping and stack.
        # self.N_FEATURES = [4, 80, 80]  # With cropping and stack.
        self.N_ACTIONS = 4
        self.IMAGE_SIZE = 80
        self.N_STACK = 4
        self.model = 'REINFORCE'

        self.MAX_EPISODES = 50001  # 50001 : 500
        self.LEARNING_RATE = 0.0001
        self.DISCOUNT_FACTOR = 0.99

        # log and output
        self.WEIGHTS_SAVER_ITER = 4000  # 4000 : 200
        self.OUTPUT_SAVER_ITER = 2000  # 2000 : 100
        self.OUTPUT_GRAPH = False
        self.SAVED_NETWORK_PATH = './logs/network/'
        self.LOGS_DATA_PATH = './logs/data/'
        self.SAVED_NETWORK_PATH_BACK = './backup/network/'
        self.LOGS_DATA_PATH_BACK = './backup/data/'
