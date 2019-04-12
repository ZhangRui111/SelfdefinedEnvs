import numpy as np
import tensorflow as tf

GRAPH_LOG_PATH = "./logs/graph"
LAYER_ONE_UNITS = 64
LAYER_TWO_UNITS = 32
FULLY_LAYER_UNITS = 10


class DeepQNetwork(object):
    """Deep Q Network off-policy

    Attributes:
        none.
    """
    def __init__(self, actions, features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_policy_threshold=0,
                 e_greedy_increment=None,
                 ):
        """Initialize  DeepQNetwork's parameters

        :param actions: list -- A list includes all actions, such as [[1, 0, 0, 0], [1, 0], [1, 0, 0]].
        :param features: int -- input features's length
        :param learning_rate:
        :param reward_decay:
        :param e_greedy:
        :param replace_target_iter: int -- interval between target network's replacement.
        :param memory_size:
        :param batch_size:
        :param e_policy_threshold: int -- threshold when to start epsilon-greedy policy.
        :param e_greedy_increment: bool/int -- whether adopt e-greedy increase strategy/the increment.
        """
        self.n_actions = actions
        self.len_n_actions = len(self.n_actions[0]) + len(self.n_actions[1]) + len(self.n_actions[2])
        self.n_features = features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.memory_counter = 0  # counter in memory pool's update.
        self.batch_size = batch_size
        self.e_policy_threshold = e_policy_threshold
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0
        self.cost_his = []
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features + self.len_n_actions + 1 + self.n_features))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # start a session
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(GRAPH_LOG_PATH, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        """Build the evaluate_net and target_net.

        :return:
        """
        # ------------------ all inputs ------------------------
        # input for target net
        self.eval_net_input = tf.placeholder(tf.float32, [None, self.n_features], name='eval_net_input')
        # input for eval net
        self.target_net_input = tf.placeholder(tf.float32, [None, self.n_features], name='target_net_input')
        # q_target for loss
        self.q_target = tf.placeholder(tf.float32, [None, self.len_n_actions], name='q_target')
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e_layer1 = tf.layers.dense(self.eval_net_input, LAYER_ONE_UNITS, tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='e_layer1')
            e_layer2 = tf.layers.dense(e_layer1, LAYER_TWO_UNITS, tf.nn.relu, kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='e_layer2')
            self.eval_fully_layer = tf.layers.dense(e_layer2, FULLY_LAYER_UNITS, kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer, name='e_fully_layer')
            self.eval_action_0_out = tf.layers.dense(self.eval_fully_layer, len(self.n_actions[0]),
                                                     kernel_initializer=w_initializer,
                                                     bias_initializer=b_initializer, name='q_e_action_0')
            self.eval_action_1_out = tf.layers.dense(self.eval_fully_layer, len(self.n_actions[1]),
                                                     kernel_initializer=w_initializer,
                                                     bias_initializer=b_initializer, name='q_e_action_1')
            self.eval_action_2_out = tf.layers.dense(self.eval_fully_layer, len(self.n_actions[2]),
                                                     kernel_initializer=w_initializer,
                                                     bias_initializer=b_initializer, name='q_e_action_2')
            self.q_eval_net_out = self.eval_action_0_out + self.eval_action_1_out + self.eval_action_2_out
            """
            self.eval_action_0_out is q value for the 1st set of actions.
            self.eval_action_1_out is q value for the 2nd set of actions.
            self.eval_action_2_out is q value for the 3rd set of actions.
            self.q_eval_net_out is a value for all actions.
            
            For example:
            self.eval_action_0_out:[6.45, 2.35, 1.58, 0.99]
            self.eval_action_1_out:[8.56, 3.57]
            self.eval_action_2_out:[9.75, 2.35, 4.68]
            self.q_eval_net_out:[6.45, 2.35, 1.58, 0.99, 8.56, 3.57, 9.75, 2.35, 4.68]
            """

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_net_out, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t_layer1 = tf.layers.dense(self.eval_net_input, LAYER_ONE_UNITS, tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='t_layer1')
            t_layer2 = tf.layers.dense(t_layer1, LAYER_TWO_UNITS, tf.nn.relu, kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='t_layer2')
            self.target_fully_layer = tf.layers.dense(t_layer2, FULLY_LAYER_UNITS, kernel_initializer=w_initializer,
                                                      bias_initializer=b_initializer, name='t_fully_layer')
            self.target_action_0_out = tf.layers.dense(self.target_fully_layer, len(self.n_actions[0]),
                                                       kernel_initializer=w_initializer,
                                                       bias_initializer=b_initializer, name='q_t_action_0')
            self.target_action_1_out = tf.layers.dense(self.target_fully_layer, len(self.n_actions[1]),
                                                       kernel_initializer=w_initializer,
                                                       bias_initializer=b_initializer, name='q_t_action_1')
            self.target_action_2_out = tf.layers.dense(self.target_fully_layer, len(self.n_actions[2]),
                                                       kernel_initializer=w_initializer,
                                                       bias_initializer=b_initializer, name='q_t_action_2')
            self.q_target_net_out = self.target_action_0_out + self.target_action_1_out + self.target_action_2_out

    def store_transition(self, s, a, r, s_):
        """Store a transition in memory and replace the old transition with new transition.

        :param s: state before transformation.
        :param a: action selected, such as [[1, 0, 0, 0], [1, 0], [1, 0, 0]]
        :param r: reward get in this transition.
        :param s_: state after transformation.
        :return:
        """
        a_flat = a[0] + a[1] + a[2]
        transition = np.hstack((s, a_flat, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, step):
        """Choose action in given state/observation following e-greedy policy.

        :param observation:
        :param step:
        :return:
        """
        # at the very beginning, only take actions randomly
        if step >= self.e_policy_threshold and np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value_0, actions_value_1, actions_value_2 = self.sess.run(
                [self.eval_action_0_out, self.eval_action_1_out, self.eval_action_2_out],
                feed_dict={self.eval_net_input: observation})
            action = [[0, 0, 0, 0], [0, 0], [0, 0, 0]]
            action_0 = np.argmax(actions_value_0)
            action_1 = np.argmax(actions_value_1)
            action_2 = np.argmax(actions_value_2)
            action[0][action_0] = 1
            action[1][action_1] = 1
            action[2][action_2] = 1
        else:
            action = [[0, 0, 0, 0], [0, 0], [0, 0, 0]]
            action_0 = np.random.randint(0, len(self.n_actions[0]))
            action_1 = np.random.randint(0, len(self.n_actions[1]))
            action_2 = np.random.randint(0, len(self.n_actions[2]))
            action[0][action_0] = 1
            action[1][action_1] = 1
            action[2][action_2] = 1
        return action

    def choose_action_eval(self, observation):
        """Choose action in given state/observation following greedy policy.

        :param observation:
        :return:
            actions_value: maximal q value.
            action: selected action.
        """
        actions_value_0, actions_value_1, actions_value_2 = self.sess.run(
            [self.eval_action_0_out, self.eval_action_1_out, self.eval_action_2_out],
            feed_dict={self.eval_net_input: observation})
        action = [[0, 0, 0, 0], [0, 0], [0, 0, 0]]
        action_0 = np.argmax(actions_value_0)
        action_1 = np.argmax(actions_value_1)
        action_2 = np.argmax(actions_value_2)
        action[0][action_0] = 1
        action[1][action_1] = 1
        action[2][action_2] = 1
        return action

    def learn(self):
        """Training and adjusting networks.

        :return:
        """
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample a batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # input is all next observation
        q_target_select_a, q_target_out = self.sess.run([self.q_eval_net_out, self.q_target_net_out],
                                                        feed_dict={
                                                            self.eval_net_input: batch_memory[:, -self.n_features:],
                                                            self.target_net_input: batch_memory[:, -self.n_features:]})
        # real q_eval, input is the current observation
        q_eval = self.sess.run(self.q_eval_net_out, {self.eval_net_input: batch_memory[:, :self.n_features]})
        tf.summary.histogram("q_eval", q_eval)

        q_target = q_eval.copy()

        # action a in transition[s, a, r, s_]
        eval_act_index = batch_memory[:, self.n_features:(self.n_features + self.len_n_actions)].astype(int)
        reward = batch_memory[:, (self.n_features + self.len_n_actions)]  # reward r in transition[s, a, r, s_]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        selected_q_next = np.max(q_target_out, axis=1)

        # q_target
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        tf.summary.histogram("q_target", q_target)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        tf.summary.scalar("cost", self.cost)

        # merge_all() must follow all tf.summary
        self.merge_op = tf.summary.merge_all()
        merge_all = self.sess.run(self.merge_op, feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                            self.q_target: q_target})
        self.writer.add_summary(merge_all, self.learn_step_counter)

        # epsilon-decay
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
