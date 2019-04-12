import cv2
import numpy as np
import tensorflow as tf

# Clears the default graph stack and resets the global default graph.
# tf.reset_default_graph()


class SumTree(object):
    """ This SumTree code is modified version and the original code is from:
        https://github.com/jaara/AI-blog/blob/master/SumTree.py
        Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        self.full = False  # whether the reply pool is full.
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, p)  # update tree_frame

        self.data[self.data_pointer] = data  # update data_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.full = True
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            # tree_idx is original tree_idx's parent node. (cl_idx = 2 * parent_idx + 1, cr_idx = cl_idx + 1)
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """ This SumTree code is modified version and the original code is from:
        https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, capacity, para):
        self.tree = SumTree(capacity)
        self._init_paras(para)

    def _init_paras(self, para):
        self.epsilon = para.epsilon  # small amount to avoid zero priority
        self.alpha = para.alpha  # [0~1] convert the importance of TD error to priority
        self.beta = para.beta  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = para.beta_increment_per_sampling
        self.abs_err_upper = para.abs_err_upper  # clipped abs error

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n, incre_deta=False):
        b_idx, b_memory, ISWeights = \
            np.empty((n, 1), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment

        if incre_deta is True:
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # beta_max = 1
        else:
            self.beta = np.min([1., self.beta])  # beta_max = 1
        
        if self.tree.full is True:
            # for later calculate ISweight, what the leaf note store is ``(p_i)^\alpha'',
            # so P(i) = (p_i)^\alpha / self.tree.total_p
            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        else:
            start = self.tree.capacity - 1
            end = self.tree.data_pointer + self.tree.capacity - 1
            min_prob = np.min(self.tree.tree[start:end]) / self.tree.total_p

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            # About ISWeights's calculation,
            # https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx
            data_ = np.asarray(data)
            b_memory[i, :] = data_
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # p_i = |\delta| + \epsilon, to avoid p_i is zero.
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        # When calculate P(i), we will need (p_i)^\alpha, or in other way, (p_i)^\alpha is a linear measurement of
        # its probability of being replayed. See P(i)'s formula in the paper.
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class MemoryParas(object):
    def __init__(self, m_nonzero_epsilon, m_alpha, m_initial_bata, m_final_bata, m_final_frame, m_abs_err_upper):
        self.epsilon = m_nonzero_epsilon  # small amount to avoid zero priority
        self.alpha = m_alpha  # [0~1] convert the importance of TD error to priority
        self.beta = m_initial_bata  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = (m_final_bata-m_initial_bata)/m_final_frame
        self.abs_err_upper = m_abs_err_upper  # clipped abs error, usually is 1.


class BaseDQN(object):
    def __init__(self,
                 hp,
                 token,
                 network_build=None,
                 prioritized=False,
                 initial_epsilon=None,
                 finial_epsilon=None,
                 finial_epsilon_frame=None,
                 discount_factor=None,
                 minibatch_size=None,
                 reply_start=None,
                 reply_memory_size=None,
                 target_network_update_frequency=None,
                 output_graph=None):

        self.hp = hp
        self.token = token
        if initial_epsilon is None:
            self.epsilon = self.hp.INITIAL_EXPLOR
        else:
            self.epsilon = initial_epsilon
        if initial_epsilon is None:
            self.epsilon_max = self.hp.FINAL_EXPLOR
        else:
            self.epsilon_max = finial_epsilon
        if finial_epsilon_frame is None:
            self.epsilon_increment = (self.epsilon_max - self.epsilon) / self.hp.FINAL_EXPLOR_FRAME
        else:
            self.epsilon_increment = (self.epsilon_max - self.epsilon) / finial_epsilon_frame
        if discount_factor is None:
            self.gamma = self.hp.DISCOUNT_FACTOR
        else:
            self.gamma = discount_factor
        if minibatch_size is None:
            self.batch_size = self.hp.MINIBATCH_SIZE
        else:
            self.batch_size = minibatch_size
        if reply_start is None:
            self.replay_start = self.hp.REPLY_START_SIZE
        else:
            self.replay_start = reply_start
        if reply_memory_size is None:
            self.memory_size = self.hp.REPLY_MEMORY_SIZE
        else:
            self.memory_size = reply_memory_size
        if target_network_update_frequency is None:
            self.replace_target_iter = self.hp.TARGET_NETWORK_UPDATE_FREQUENCY
        else:
            self.replace_target_iter = target_network_update_frequency
        if output_graph is None:
            self.summary_flag = self.hp.OUTPUT_GRAPH
        else:
            self.summary_flag = output_graph

        self.n_actions = self.hp.N_ACTIONS
        self.n_stack = self.hp.N_STACK
        self.image_size = self.hp.IMAGE_SIZE
        self.max_episode = self.hp.MAX_EPISODES
        self.prioritized = prioritized
        self.flag = True  # output signal

        # network input/output
        if network_build is not None:
            self.eval_net_input = network_build[0][0]
            self.target_net_input = network_build[0][1]
            self.q_target = network_build[0][2]
            self.q_eval_net_out = network_build[1][0]
            self.loss = network_build[1][1]
            self.q_target_net_out = network_build[1][2]
            self.e_params = network_build[2][0]
            self.t_params = network_build[2][1]
            self.train_op = network_build[2][2]

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        if self.prioritized:
            memory_paras = MemoryParas(self.hp.M_NONZERO_EPSILON, self.hp.M_ALPHA, self.hp.M_INITIAL_BETA, self.hp.M_FINAL_BETA,
                                       self.hp.M_FINAL_BETA_FRAME, self.hp.M_ABS_ERROR_UPPER)
            self.memory = Memory(capacity=self.memory_size, para=memory_paras)
        else:
            self.memory = []

        # target network's soft_replacement
        if hasattr(self, 't_params'):
            with tf.variable_scope('soft_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        # start a session
        gpu_options = tf.GPUOptions(allow_growth=True)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.graph_path = self.hp.LOGS_DATA_PATH + self.hp.model + '/' + self.token + '/'

        if self.summary_flag:
            self.writer = tf.summary.FileWriter(self.graph_path, self.sess.graph)

        # self.cost_his = []

    def preprocess_image(self, img):
        img = img[30:-15, 5:-5:, :]  # image cropping
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAY
        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return gray

    def store_transition(self, s, a, r, s_):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_))

    def choose_action(self, observation):
        """ Choose action following epsilon-greedy policy.
        """
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval_net_out, feed_dict={
                self.eval_net_input: observation.reshape(1, self.n_stack, self.image_size, self.image_size)})
            action_index = np.argmax(actions_value)
        else:
            action_index = np.random.randint(0, self.n_actions)
        action = action_index
        return action
        # my_print(action, '-')
        # filename = self.hp.LOGS_DATA_PATH + self.hp.model + '/' + 'log_details.txt'
        # write_file(filename, str(action))

    def choose_action_greedy(self, observation):
        """ Choose action following greedy policy.
        """
        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
        action_index = np.argmax(actions_value)
        action = action_index
        return action

    def learn(self, incre_epsilon):
        pass

    def close_session(self):
        self.sess.close()
