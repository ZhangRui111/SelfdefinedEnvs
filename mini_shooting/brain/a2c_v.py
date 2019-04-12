import cv2
import math
import numpy as np
import random

from brain.base_dqns import BaseDQN
from hyper_paras.hp_a2c_v import Hyperparameters
from shared.utils import write_file


class Actor(object):
    def __init__(self, sess, network=None):
        self.sess = sess
        self.hp = Hyperparameters()

        # input placeholder
        self.state = network[0][0]
        # self.action = network[0][1]
        self.td_error = network[0][1]
        # output
        self.acts_prob = network[1][0]
        self.exp_v = network[1][1]
        self.train_op = network[1][2]

    def learn(self, s, a, td_error):
        one_hot_a = np.eye(self.hp.N_ACTIONS)[a]
        one_hot_td_error = td_error * one_hot_a
        _, exp_v = self.sess.run([self.train_op, self.exp_v],
                                 feed_dict={self.state: s,
                                            self.td_error: one_hot_td_error})
        # *.reshape(-1) is necessary, or cannot feed (32, 1) to placeholder which has shape (32,)
        if math.isnan(exp_v) is True:
            print('nan for exp_v')
            raise Exception("nan error exp_v")

        return exp_v

    def choose_action(self, observation):
        # get probabilities for all actions
        probs = self.sess.run(self.acts_prob, {self.state: observation[np.newaxis, :]})
        select_action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return select_action, probs  # return a int


class Critic(object):
    def __init__(self, sess, network=None):
        self.sess = sess
        self.hp = Hyperparameters()

        # input placeholder
        self.state = network[0][0]
        self.target_v = network[0][1]
        # self.next_value = network[0][1]
        # self.reward = network[0][2]
        # output
        self.value = network[1][0]
        # self.td_error = network[1][1]
        self.loss = network[1][1]
        self.train_op = network[1][2]

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.value, {self.state: s_})
        v = self.sess.run(self.value, {self.state: s})
        td_target = r[:, np.newaxis] + self.hp.DISCOUNT_FACTOR * v_
        td_error = td_target - v
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.state: s,
                                           self.target_v: td_target})

        if np.sum(np.isnan(td_target)) >= 1:
            print('nan: {}'.format(np.sum(np.isnan(td_target))))
            raise Exception("nan error")

        return td_error, loss


class A2C(BaseDQN):
    def __init__(self,
                 hp,
                 token,
                 network_actor,
                 network_critic,
                 prioritized=False,
                 initial_epsilon=None,
                 finial_epsilon=None,
                 finial_epsilon_frame=None,
                 discount_factor=None,
                 minibatch_size=None,
                 reply_start=None,
                 reply_memory_size=None,
                 target_network_update_frequency=None):
        super().__init__(hp,
                         token,
                         None,  # network_build=None
                         prioritized,
                         initial_epsilon,
                         finial_epsilon,
                         finial_epsilon_frame,
                         discount_factor,
                         minibatch_size,
                         reply_start,
                         reply_memory_size,
                         target_network_update_frequency)
        self.network_actor = network_actor
        self.network_critic = network_critic
        self.actor = Actor(self.sess, network=self.network_actor)
        self.critic = Critic(self.sess, network=self.network_critic)

        write_file(self.graph_path + 'loss_exp_v_v.txt', 'critic loss: | actor exp_v:\n', True)

    def learn(self, incre_epsilon):
        self.learn_step_counter += 1

        # sample batch memory from all memory
        # zip(): Take iterable objects as parameters, wrap the corresponding elements in the object into tuples,
        # and then return a list of those tuples
        samples_batch = random.sample(self.memory, self.batch_size)  # list of tuples
        observation, eval_act_index, reward, observation_ = zip(*samples_batch)  # tuple of lists

        observation = np.array(observation)
        action = np.array(eval_act_index)
        reward = np.array(reward)
        observation_ = np.array(observation_)

        td_error, loss = self.critic.learn(observation, reward, observation_)
        exp_v = self.actor.learn(observation, action, td_error)

        # print('critic loss: {0} | actor exp_v: {1}'.format(loss, exp_v))
        content = '{0} | {1}\n'.format(loss, exp_v)
        write_file(self.graph_path + 'loss_exp_v_v.txt', content, False)

    def preprocess_image(self, img):
        img = img[30:-15, 5:-5:, :]  # image cropping
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAY
        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return gray
