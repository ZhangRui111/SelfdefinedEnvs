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

    def learn(self, incre_epsilon):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0 and self.learn_step_counter != 0:
            self.sess.run(self.target_replace_op)
            my_print('target_params_replaced', '-')

        self.learn_step_counter += 1

        # sample batch memory from all memory
        # zip(): Take iterable objects as parameters, wrap the corresponding elements in the object into tuples,
        # and then return a list of those tuples
        samples_batch = random.sample(self.memory, self.batch_size)  # list of tuples
        observation, eval_act_index, reward, observation_ = zip(*samples_batch)  # tuple of lists

        observation = np.array(observation)
        eval_act_index = np.array(eval_act_index)
        reward = np.array(reward)
        observation_ = np.array(observation_)

        # input is all next observation
        q_eval_input_s_next, q_target_input_s_next = \
            self.sess.run([self.q_eval_net_out, self.q_target_net_out], feed_dict={
                self.eval_net_input: observation_.reshape((-1, self.n_stack, self.image_size, self.image_size)),
                self.target_net_input: observation_.reshape((-1, self.n_stack, self.image_size, self.image_size))})
        # real q_eval, input is the current observation
        q_eval_input_s = self.sess.run(self.q_eval_net_out, feed_dict={
            self.eval_net_input: observation.reshape((-1, self.n_stack, self.image_size, self.image_size))})

        if self.summary_flag:
            tf.summary.histogram("q_eval", q_eval_input_s)

        # q target
        q_target = q_eval_input_s.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        max_act_next = np.argmax(q_eval_input_s_next, axis=1)
        selected_q_next = q_target_input_s_next[batch_index, max_act_next]
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={
            self.eval_net_input: observation.reshape((-1, self.n_stack, self.image_size, self.image_size)),
            self.q_target: q_target})

        # epsilon-decay
        if incre_epsilon:
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
