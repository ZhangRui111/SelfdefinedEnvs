"""
This is the prioritized reply dqn with proportional stochastic prioritization.
Another stochastic prioritization way is rank-based stochastic prioritization.
"""
import numpy as np
import random
import tensorflow as tf

from brain.base_dqns import BaseDQN
from shared.utils import my_print

# Clears the default graph stack and resets the global default graph.
# tf.reset_default_graph()


class DeepQNetwork(BaseDQN):
    def __init__(self,
                 hp,
                 token,
                 network_build,
                 prioritized=True,
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
                         network_build,
                         prioritized,
                         initial_epsilon,
                         finial_epsilon,
                         finial_epsilon_frame,
                         discount_factor,
                         minibatch_size,
                         reply_start,
                         reply_memory_size,
                         target_network_update_frequency)
        self.abs_error = network_build[3][0]
        self.ISWeights = network_build[3][1]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.flatten(), a, [r], s_.flatten()))
        self.memory.store(transition)  # have high priority for newly arrived transition

    def learn(self, episode_done):
        """
        :param episode_done: only update hyper-parameter beta (in pri_dqn) and epsilon when episode_done is True.
        :return:
        """
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0 and self.learn_step_counter != 0:
            self.sess.run(self.target_replace_op)
            my_print('target_params_replaced', '-')

        self.learn_step_counter += 1

        # sample batch, tree_idx is not used.
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size, episode_done)

        length = self.n_stack * self.image_size * self.image_size
        observation = batch_memory[:, :length]
        eval_act_index = batch_memory[:, length].astype(int)
        reward = batch_memory[:, length + 1]
        observation_ = batch_memory[:, -length:]

        # input is all next observation
        q_target_input_s_next = self.sess.run(self.q_target_net_out, feed_dict={
            self.target_net_input: observation_.reshape((-1, self.n_stack, self.image_size, self.image_size))})
        # real q_eval, input is the current observation
        q_eval_input_s = self.sess.run(self.q_eval_net_out, feed_dict={
            self.eval_net_input: observation.reshape((-1, self.n_stack, self.image_size, self.image_size))})

        if self.summary_flag:
            tf.summary.histogram("q_eval", q_eval_input_s)

        # q target
        q_target = q_eval_input_s.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        selected_q_next = np.max(q_target_input_s_next, axis=1)
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, abs_errors, cost = self.sess.run([self.train_op, self.abs_error, self.loss], feed_dict={
            self.eval_net_input: observation.reshape((-1, self.n_stack, self.image_size, self.image_size)),
            self.q_target: q_target,
            self.ISWeights: ISWeights})

        # epsilon-decay
        if episode_done:
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        if self.summary_flag:
            tf.summary.scalar("cost", cost)
            # merge_all() must follow all tf.summary
            if self.flag:
                self.merge_op = tf.summary.merge_all()
                self.flag = False
            merge_all = self.sess.run(self.merge_op, feed_dict={
                self.eval_net_input: observation.reshape((-1, self.n_stack, self.image_size, self.image_size)),
                self.q_target: q_target})
            self.writer.add_summary(merge_all, self.learn_step_counter)
