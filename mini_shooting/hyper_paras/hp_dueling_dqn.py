from hyper_paras.base_hyper_paras import BaseHyperparameters


class Hyperparameters(BaseHyperparameters):
    def __init__(self):
        super().__init__()
        self.model = 'dueling_dqn'
