from hyper_paras.base_hyper_paras import BaseHyperparameters


class Hyperparameters(BaseHyperparameters):
    def __init__(self):
        super().__init__()
        self.model = 'A2C_V'

        self.MAX_EPISODES = 50001  # 50001 : 500
        self.LEARNING_RATE_ACTOR = 0.00005
        self.LEARNING_RATE_CRITIC = 0.0001
        self.DISCOUNT_FACTOR = 0.9
        self.OUTPUT_SAVER_ITER = 500

        # for test.
        # self.OUTPUT_SAVER_ITER = 20
        # self.REPLY_START_SIZE = 20
        # self.MINIBATCH_SIZE = 8
