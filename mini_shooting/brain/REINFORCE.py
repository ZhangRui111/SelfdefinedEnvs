import cv2
import numpy as np
import tensorflow as tf


class REINFORCE:
    def __init__(
            self,
            network_build,
            hp,
            token,
            discount_factor=None,
            output_graph=False,
    ):

        self.hp = hp
        self.token = token
        if discount_factor is None:
            self.gamma = self.hp.DISCOUNT_FACTOR
        else:
            self.gamma = discount_factor
        if output_graph is None:
            self.summary_flag = self.hp.OUTPUT_GRAPH
        else:
            self.summary_flag = output_graph
        self.n_actions = self.hp.N_ACTIONS
        self.n_stack = self.hp.N_STACK
        self.image_size = self.hp.IMAGE_SIZE
        self.max_episode = self.hp.MAX_EPISODES
        self.flag = True  # output signal

        # network input/output
        self.tf_observation = network_build[0][0]
        self.tf_actions = network_build[0][1]
        self.tf_vt = network_build[0][2]
        self.all_act_prob = network_build[1][0]
        self.loss = network_build[1][1]
        self.train_op = network_build[1][2]

        # total learning step
        self.learn_step_counter = 0

        # initialize episode memory to store states, actions, rewards
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # start a session
        gpu_options = tf.GPUOptions(allow_growth=True)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.graph_path = self.hp.LOGS_DATA_PATH + self.hp.model + '/' + self.token + '/'

        if self.summary_flag:
            self.writer = tf.summary.FileWriter(self.graph_path, self.sess.graph)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_observation: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, prob_weights

    def store_transition(self, s, a, r, s_):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             # shape=[episode_size, n_stack, image_size, image_size]
             self.tf_observation: np.vstack(self.ep_obs).reshape((-1, self.n_stack, self.image_size, self.image_size)),
             self.tf_actions: np.array(self.ep_as),  # shape=[episode_size, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[episode_size, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        # This way we’re always encouraging and discouraging roughly half of the performed actions.
        mean = np.mean(discounted_ep_rs)
        std = np.std(discounted_ep_rs) + 1E-6
        discounted_ep_rs = (discounted_ep_rs-mean)/std
        return discounted_ep_rs

    def preprocess_image(self, img):
        img = img[30:-15, 5:-5:, :]  # image cropping
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAY
        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return gray
