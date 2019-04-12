import tensorflow as tf
from hyper_paras.hp_dqn_2015 import Hyperparameters


def build_network(lr=None, n_stack=None, image_size=None, n_actions=None):
    """ Build the network for RL algorithm.
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
    # # connect None with n_features or n_actions.
    # if type(n_features) is list:
    #     f = n_features.copy()
    #     f.insert(0, None)
    #     features = f
    # else:
    #     features = [None] + n_features
    # if type(n_actions) is list:
    #     a = n_actions.copy()
    #     a.insert(0, None)
    #     actions = a
    # else:
    #     actions = [None] + [n_actions]

    """This network occupy 879Mib GPU memory."""
    # ------------------ all inputs --------------------------
    # input for target net
    eval_net_input = tf.placeholder(tf.float32, shape=[None, n_stack, image_size, image_size],
                                    name='eval_net_input_' + flag)
    # input for eval net
    target_net_input = tf.placeholder(tf.float32, shape=[None, n_stack, image_size, image_size],
                                      name='target_net_input_' + flag)
    # q_target for loss
    q_target = tf.placeholder(tf.float32, shape=[None, n_actions], name='q_target_' + flag)

    # ------------------ build evaluate_net ------------------
    with tf.variable_scope('eval_net_' + flag):
        # (?, 4, 9, 9)
        e_input_crop = eval_net_input / 9
        e_input = tf.transpose(e_input_crop, [0, 2, 3, 1])  # (?, 9, 9, 4)
        # tf.contrib.layers.conv2d(..., activation_fn=tf.nn.relu,...)
        e_conv1 = tf.contrib.layers.conv2d(inputs=e_input, num_outputs=16, kernel_size=3, stride=1)  # (?, 7, 7, 16)
        e_conv2 = tf.contrib.layers.conv2d(inputs=e_conv1, num_outputs=32, kernel_size=3, stride=1)  # (?, 5, 5, 32)

        e_flat = tf.contrib.layers.flatten(e_conv2)
        e_f = tf.contrib.layers.fully_connected(e_flat, 512)
        e_out = tf.contrib.layers.fully_connected(e_f, n_actions)

    with tf.variable_scope('loss_' + flag):
        loss = tf.reduce_mean(tf.squared_difference(q_target, e_out, name='TD_error_' + flag))

    with tf.variable_scope('train_' + flag):
        _train_op = tf.train.RMSPropOptimizer(lr, 0.99, 0.0, 1e-6).minimize(loss)

    # ------------------ build target_net --------------------
    with tf.variable_scope('target_net_' + flag):
        # (?, 4, 80, 80)
        # (?, 4, 9, 9)
        t_input_crop = target_net_input / 9
        t_input = tf.transpose(t_input_crop, [0, 2, 3, 1])  # (?, 9, 9, 4)
        # tf.contrib.layers.conv2d(..., activation_fn=tf.nn.relu,...)
        t_conv1 = tf.contrib.layers.conv2d(inputs=t_input, num_outputs=16, kernel_size=3, stride=1)  # (?, 7, 7, 16)
        t_conv2 = tf.contrib.layers.conv2d(inputs=t_conv1, num_outputs=32, kernel_size=3, stride=1)  # (?, 5, 5, 32)

        t_flat = tf.contrib.layers.flatten(t_conv2)
        t_f = tf.contrib.layers.fully_connected(t_flat, 512)
        t_out = tf.contrib.layers.fully_connected(t_f, n_actions)

    """This network occupy more than 8000Mib GPU memory."""
    # # ------------------ all inputs --------------------------
    # # input for target net
    # eval_net_input = tf.placeholder(tf.float32, features, name='eval_net_input_' + flag)
    # # input for eval net
    # target_net_input = tf.placeholder(tf.float32, features, name='target_net_input_' + flag)
    # # q_target for loss
    # q_target = tf.placeholder(tf.float32, actions, name='q_target_' + flag)
    # # initializer
    # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0)
    #
    # # ------------------ build evaluate_net ------------------
    # with tf.variable_scope('eval_net_' + flag):
    #     e_conv1 = tf.layers.conv2d(eval_net_input, filters=128, kernel_size=(3, 3), strides=(1, 1),
    #                                padding='same', activation=tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='e_conv1_' + flag)
    #     e_pool1 = tf.layers.max_pooling2d(e_conv1, (2, 2), (2, 2), padding='same', name='e_pool1_' + flag)
    #     e_conv2 = tf.layers.conv2d(e_pool1, filters=64, kernel_size=(3, 3), strides=(1, 1),
    #                                padding='same', activation=tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='e_conv2_' + flag)
    #     e_pool2 = tf.layers.max_pooling2d(e_conv2, (2, 2), (2, 2), padding='same', name='e_pool2_' + flag)
    #     e_conv3 = tf.layers.conv2d(e_pool2, filters=32, kernel_size=(3, 3), strides=(1, 1),
    #                                padding='same', activation=tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='e_conv3_' + flag)
    #     e_pool3 = tf.layers.max_pooling2d(e_conv3, (2, 2), (2, 2), padding='same', name='e_pool2_' + flag)
    #     e_flat = tf.contrib.layers.flatten(e_pool3)
    #     e_f1 = tf.layers.dense(e_flat, 128, tf.nn.relu, kernel_initializer=w_initializer,
    #                            bias_initializer=b_initializer, name='e_f1_' + flag)
    #     if two_fl is True:
    #         e_f2 = tf.layers.dense(e_f1, 64, tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='e_f2_' + flag)
    #         e_out = tf.layers.dense(e_f2, n_actions, kernel_initializer=w_initializer,
    #                                 bias_initializer=b_initializer, name='q_e_' + flag)
    #     else:
    #         e_out = tf.layers.dense(e_f1, n_actions, kernel_initializer=w_initializer,
    #                                 bias_initializer=b_initializer, name='q_e_' + flag)
    #
    # with tf.variable_scope('loss_' + flag):
    #     loss = tf.reduce_mean(tf.squared_difference(q_target, e_out, name='TD_error_' + flag))
    #
    # with tf.variable_scope('train_' + flag):
    #     _train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)
    #
    # # ------------------ build target_net --------------------
    # with tf.variable_scope('target_net_' + flag):
    #     t_conv1 = tf.layers.conv2d(target_net_input, filters=128, kernel_size=(3, 3), strides=(1, 1),
    #                                padding='same', activation=tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='t_conv1_' + flag)
    #     t_pool1 = tf.layers.max_pooling2d(t_conv1, (2, 2), (2, 2), padding='same', name='t_pool1_' + flag)
    #     t_conv2 = tf.layers.conv2d(t_pool1, filters=64, kernel_size=(3, 3), strides=(1, 1),
    #                                padding='same', activation=tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='t_conv2_' + flag)
    #     t_pool2 = tf.layers.max_pooling2d(t_conv2, (2, 2), (2, 2), padding='same', name='t_pool2_' + flag)
    #     t_conv3 = tf.layers.conv2d(t_pool2, filters=32, kernel_size=(3, 3), strides=(1, 1),
    #                                padding='same', activation=tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='t_conv3_' + flag)
    #     t_pool3 = tf.layers.max_pooling2d(t_conv3, (2, 2), (2, 2), padding='same', name='t_pool3_' + flag)
    #     t_flat = tf.contrib.layers.flatten(t_pool3)
    #     t_f1 = tf.layers.dense(t_flat, 128, tf.nn.relu, kernel_initializer=w_initializer,
    #                            bias_initializer=b_initializer, name='t_f1_' + flag)
    #     if two_fl is True:
    #         t_f2 = tf.layers.dense(t_f1, 64, tf.nn.relu, kernel_initializer=w_initializer,
    #                                bias_initializer=b_initializer, name='t_f2_' + flag)
    #         t_out = tf.layers.dense(t_f2, n_actions, kernel_initializer=w_initializer,
    #                                 bias_initializer=b_initializer, name='t_e_' + flag)
    #     else:
    #         t_out = tf.layers.dense(t_f1, n_actions, kernel_initializer=w_initializer,
    #                                 bias_initializer=b_initializer, name='t_e_' + flag)

    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_' + flag)
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_' + flag)

    return [[eval_net_input, target_net_input, q_target],
            [e_out, loss, t_out],
            [e_params, t_params, _train_op]]


if __name__ == '__main__':
    build_network(0.0001)
