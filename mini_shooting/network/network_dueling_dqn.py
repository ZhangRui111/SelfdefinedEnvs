import tensorflow as tf
from hyper_paras.hp_dueling_dqn import Hyperparameters


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
        # (?, 4, 80, 80)
        e_input_crop = eval_net_input / 255
        e_input = tf.transpose(e_input_crop, [0, 2, 3, 1])  # (?, 80, 80, 4)
        # tf.contrib.layers.conv2d(..., activation_fn=tf.nn.relu,...)
        e_conv1 = tf.contrib.layers.conv2d(inputs=e_input, num_outputs=32, kernel_size=8, stride=4)  # (?, 20, 20, 32)
        e_conv2 = tf.contrib.layers.conv2d(inputs=e_conv1, num_outputs=64, kernel_size=4, stride=2)  # (?, 10, 10, 64)
        e_conv3 = tf.contrib.layers.conv2d(inputs=e_conv2, num_outputs=64, kernel_size=3, stride=1)  # (?, 10, 10, 64)

        e_flat = tf.contrib.layers.flatten(e_conv3)
        e_f = tf.contrib.layers.fully_connected(e_flat, 512)

        eval_V = tf.contrib.layers.fully_connected(e_f, 1)
        eval_A = tf.contrib.layers.fully_connected(e_f, n_actions)

        e_out = eval_V + (eval_A - tf.reduce_mean(eval_A, axis=1, keepdims=True))

    with tf.variable_scope('loss_' + flag):
        loss = tf.reduce_mean(tf.squared_difference(q_target, e_out, name='TD_error_' + flag))

    with tf.variable_scope('train_' + flag):
        _train_op = tf.train.RMSPropOptimizer(lr, 0.99, 0.0, 1e-6).minimize(loss)

    # ------------------ build target_net --------------------
    with tf.variable_scope('target_net_' + flag):
        # (?, 4, 80, 80)
        t_input_crop = target_net_input / 255
        t_input = tf.transpose(t_input_crop, [0, 2, 3, 1])  # (?, 80, 80, 4)
        # tf.contrib.layers.conv2d(..., activation_fn=tf.nn.relu,...)
        t_conv1 = tf.contrib.layers.conv2d(inputs=t_input, num_outputs=32, kernel_size=8, stride=4)  # (?, 20, 20, 32)
        t_conv2 = tf.contrib.layers.conv2d(inputs=t_conv1, num_outputs=64, kernel_size=4, stride=2)  # (?, 10, 10, 64)
        t_conv3 = tf.contrib.layers.conv2d(inputs=t_conv2, num_outputs=64, kernel_size=3, stride=1)  # (?, 10, 10, 64)

        t_flat = tf.contrib.layers.flatten(t_conv3)
        t_f = tf.contrib.layers.fully_connected(t_flat, 512)
        target_V = tf.contrib.layers.fully_connected(t_f, 1)
        target_A = tf.contrib.layers.fully_connected(t_f, n_actions)

        t_out = target_V + (target_A - tf.reduce_mean(target_A, axis=1, keepdims=True))

    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_' + flag)
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_' + flag)

    return [[eval_net_input, target_net_input, q_target],
            [e_out, loss, t_out],
            [e_params, t_params, _train_op]]


if __name__ == '__main__':
    build_network(0.0001)
