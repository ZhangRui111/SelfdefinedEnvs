import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.

REPLY_START_SIZE = 50000
LOGS_DATA_PATH = './logs_data/hard_maze/dqn_rvnet/'
LOGS_EVENTS_PATH = './logs_events/hard_maze/dqn_rvnet/'


class DQNBrainRvNet(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.flag = True  # tf.summary.merge
        self.summary_flag = output_graph  # tf.summary flag

        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 1 + len(self.n_actions)))

        # consist of [target_net, evaluate_net]
        self.create_network()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # start a session
        self.sess = tf.Session()

        if self.summary_flag:
            self.writer = tf.summary.FileWriter(LOGS_EVENTS_PATH, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        # self.cost_his = []

    def create_network(self):
        # ------------------ all inputs ------------------------
        # input for target net
        self.eval_net_input = tf.placeholder(tf.float32, [None, self.n_features], name='eval_net_input')
        # input for eval net
        self.target_net_input = tf.placeholder(tf.float32, [None, self.n_features], name='target_net_input')
        # q_target for loss
        self.q_target = tf.placeholder(tf.float32, [None, sum(self.n_actions)], name='q_target')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.eval_net_input, 256, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e2, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            """The net structure still need to be modified to adapt more kinds of action"""
            em1 = tf.layers.dense(e3, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer, name='em1')
            q_action0 = tf.layers.dense(em1, self.n_actions[0], kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='q_a0')
            # u1 = np.hstack((e2, q_action0))
            eu1 = tf.concat([e3, q_action0], 1)
            em2 = tf.layers.dense(eu1, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer, name='em2')
            q_action1 = tf.layers.dense(em2, self.n_actions[1], kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='q_a1')
            # self.q_eval_net_out_d = tf.layers.dense(e2, self.n_actions[0], kernel_initializer=w_initializer,
            #                                         bias_initializer=b_initializer, name='q_d')
            # self.q_eval_net_out_j = tf.layers.dense(e2, self.n_actions[1], kernel_initializer=w_initializer,
            #                                         bias_initializer=b_initializer, name='q_j')
            # self.q_eval_net_out = tf.layers.dense(e2, sum(self.n_actions), kernel_initializer=w_initializer,
            #                                       bias_initializer=b_initializer, name='q')
            self.q_eval_net_out = tf.concat([q_action0, q_action1], 1)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_net_out, name='TD_error'))
            if self.summary_flag:
                tf.summary.scalar("loss", self.loss)

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.target_net_input, 256, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dense(t2, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            """The net structure still need to be modified to adapt more kinds of action"""
            tm1 = tf.layers.dense(t3, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer, name='tm1')
            t_action0 = tf.layers.dense(tm1, self.n_actions[0], kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='t_a0')
            # u1 = np.hstack((t2, t_action0))
            tu1 = tf.concat([t3, t_action0], 1)
            tm2 = tf.layers.dense(tu1, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer, name='tm2')
            t_action1 = tf.layers.dense(tm2, self.n_actions[1], kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='t_a1')
            # self.q_target_net_out = tf.layers.dense(t2, sum(self.n_actions), kernel_initializer=w_initializer,
            #                                         bias_initializer=b_initializer, name='t')
            self.q_target_net_out = tf.concat([t_action0, t_action1], 1)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        a_ = convert_action_to_int(a, len(self.n_actions))
        transition = np.hstack((s.flatten(), a_, [r], s_.flatten()))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, step):
        # at the very beginning, only take actions randomly
        action = [0] * len(self.n_actions)
        if step < REPLY_START_SIZE:
            action[0] = np.random.randint(0, self.n_actions[0])
            action[1] = np.random.randint(self.n_actions[0], sum(self.n_actions))
        else:
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
                action[0] = np.argmax(actions_value[0][:self.n_actions[0]])
                action[1] = np.argmax(actions_value[0][self.n_actions[0]:sum(self.n_actions)])
            else:
                action[0] = np.random.randint(0, self.n_actions[0])
                action[1] = np.random.randint(self.n_actions[0], sum(self.n_actions))
        actions_ = convert_action_to_string(action, len(self.n_actions))
        return actions_

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # input is all next observation
        q_target_select_a, q_target_out = \
            self.sess.run([self.q_eval_net_out, self.q_target_net_out],
                          feed_dict={self.eval_net_input: batch_memory[:, -self.n_features:],
                                     self.target_net_input: batch_memory[:, -self.n_features:]})
        # real q_eval, input is the current observation.
        q_eval = self.sess.run(self.q_eval_net_out, {self.eval_net_input: batch_memory[:, :self.n_features]})
        if self.summary_flag:
            tf.summary.histogram("q_eval", q_eval)

        # self.eval_ddqn = q_eval

        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # # Double DQN
        # max_act4next = np.argmax(q_target_select_a, axis=1)
        # selected_q_next = q_target_out[batch_index, max_act4next]
        # # DQN
        selected_q_next = np.max(q_target_out, axis=1)

        # real q_target
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        if self.summary_flag:
            tf.summary.histogram("q_target", q_target)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # self.cost_his.append(self.cost)
        # tf.summary.scalar("cost", self.cost)

        if self.summary_flag:
            if self.flag:
                # merge_all() must follow all tf.summary
                self.merge_op = tf.summary.merge_all()
                self.flag = False
            merge_all = self.sess.run(self.merge_op, feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                                self.q_target: q_target})
            self.writer.add_summary(merge_all, self.learn_step_counter)

        # epsilon-decay
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


def store_parameters(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(LOGS_EVENTS_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0

    return saver, step


def convert_action_to_string(action, size):
    actions_ = [0] * size
    if action[0] == 0:
        actions_[0] = 'w'
    elif action[0] == 1:
        actions_[0] = 's'
    elif action[0] == 2:
        actions_[0] = 'a'
    else:
        actions_[0] = 'd'
    if action[1] == 4:
        actions_[1] = 'j'
    else:
        actions_[1] = 'n'
    return actions_


def convert_action_to_int(action, size):
    actions_ = [0] * size
    if action[0] == 'w':
        actions_[0] = 0
    elif action[0] == 's':
        actions_[0] = 1
    elif action[0] == 'a':
        actions_[0] = 2
    else:
        actions_[0] = 3
    if action[1] == 'j':
        actions_[1] = 4
    else:
        actions_[1] = 5
    return actions_


def main():
    print("This is hard_maze_dql.py")


if __name__ == '__main__':
    main()
