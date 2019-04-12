import tensorflow as tf
from hyper_paras.hp_a2c_v import Hyperparameters


def build_actor_network(lr=None, n_stack=None, image_size=None, n_actions=None):
    """ Build the network for actor.
        """
    # init Hp
    hp = Hyperparameters()
    flag = hp.model
    if lr is None:
        lr = hp.LEARNING_RATE
    if n_stack is None:
        n_stack = hp.N_STACK
    if image_size is None:
        image_size = hp.IMAGE_SIZE
    if n_actions is None:
        n_actions = hp.N_ACTIONS

    state = tf.placeholder(tf.float32, [None, n_stack, image_size, image_size], 'state_'+flag)
    # action = tf.placeholder(tf.int32, [None, ], 'act_'+flag)
    td_error = tf.placeholder(tf.float32, [None, n_actions], 'td_error_'+flag)  # TD_error

    with tf.variable_scope('Actor'):
        input_crop = state / 255
        input = tf.transpose(input_crop, [0, 2, 3, 1])  # (?, 80, 80, 4)

        # tf.contrib.layers.conv2d(..., activation_fn=tf.nn.relu,...)
        conv1 = tf.contrib.layers.conv2d(inputs=input, num_outputs=32, kernel_size=8, stride=4)  # (?, 20, 20, 32)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2)  # (?, 10, 10, 64)
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, stride=1)  # (?, 10, 10, 64)

        flat = tf.contrib.layers.flatten(conv3)
        f = tf.contrib.layers.fully_connected(flat, 512)
        acts_prob = tf.contrib.layers.fully_connected(f, n_actions, activation_fn=tf.nn.softmax)

    with tf.variable_scope('exp_v_'+flag):
        # prob = -tf.log(tf.clip_by_value(acts_prob, 1e-10, 1.0)) * tf.one_hot(action, n_actions)
        # log_prob = tf.reduce_sum(prob, axis=1)
        # exp_v = tf.reduce_mean(log_prob * td_error)  # advantage (TD_error) guided loss
        log_prob = tf.log(tf.clip_by_value(acts_prob, 1e-5, 1.0))
        exp_v = tf.reduce_mean(tf.reduce_sum(log_prob * td_error, axis=1))  # advantage (TD_error) guided loss

    with tf.variable_scope('train_'+flag):
        train_op = tf.train.AdamOptimizer(lr).minimize(-exp_v)  # minimize(-exp_v) = maximize(exp_v)

    return [[state, td_error],
            [acts_prob, exp_v, train_op]]


def build_critic_network(lr=None, n_stack=None, image_size=None, n_actions=None):
    """ Build the network for critic.
        """
    # init Hp
    hp = Hyperparameters()
    flag = hp.model
    if lr is None:
        lr = hp.LEARNING_RATE
    if n_stack is None:
        n_stack = hp.N_STACK
    if image_size is None:
        image_size = hp.IMAGE_SIZE
    if n_actions is None:
        n_actions = hp.N_ACTIONS

    state = tf.placeholder(tf.float32, [None, n_stack, image_size, image_size], 'state_' + flag)
    td_target = tf.placeholder(tf.float32, [None, 1], 't_value_' + flag)
    # next_value = tf.placeholder(tf.float32, [None, 1], 'v_next_' + flag)
    # reward = tf.placeholder(tf.float32, [None, 1], 'r_' + flag)

    with tf.variable_scope('Critic_'+flag):
        input_crop = state / 255
        input = tf.transpose(input_crop, [0, 2, 3, 1])  # (?, 80, 80, 4)

        # tf.contrib.layers.conv2d(..., activation_fn=tf.nn.relu,...)
        conv1 = tf.contrib.layers.conv2d(inputs=input, num_outputs=32, kernel_size=8, stride=4)  # (?, 20, 20, 32)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2)  # (?, 10, 10, 64)
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, stride=1)  # (?, 10, 10, 64)

        flat = tf.contrib.layers.flatten(conv3)
        f = tf.contrib.layers.fully_connected(flat, 512)
        value = tf.contrib.layers.fully_connected(f, 1, activation_fn=None)

    with tf.variable_scope('squared_TD_error_'+flag):
        # TD_error = (r+gamma*V_next) - V_eval
        # td_error = reward + hp.DISCOUNT_FACTOR * next_value - value
        loss = tf.reduce_mean(tf.squared_difference(value, td_target))

    with tf.variable_scope('train_'+flag):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    return [[state, td_target],
            [value, loss, train_op]]


if __name__ == '__main__':
    build_actor_network()
    build_critic_network()
