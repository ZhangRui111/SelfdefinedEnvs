from hyper_paras.base_hyper_paras import BaseHyperparameters


class Hyperparameters(BaseHyperparameters):
    def __init__(self):
        super().__init__()
        self.model = 'dqn_2015'

        self.MAX_EPISODES = 1000001
        self.LEARNING_RATE = 0.00001
        self.FINAL_EXPLOR_FRAME = 100000
        self.REPLY_START_SIZE = 5000
        self.REPLY_MEMORY_SIZE = 200000
        self.TARGET_NETWORK_UPDATE_FREQUENCY = 5000
