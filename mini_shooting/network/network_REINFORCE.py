import tensorflow as tf
from hyper_paras.hp_REINFORCE import Hyperparameters


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

    with tf.name_scope('inputs_'+flag):
        tf_observation = tf.placeholder(tf.float32, shape=[None, n_stack, image_size, image_size],
                                        name='observations_'+flag)
        tf_actions = tf.placeholder(tf.int32, shape=[None, ], name='actions_num_'+flag)
        tf_vt = tf.placeholder(tf.float32, shape=[None, ], name='actions_value_'+flag)

    with tf.name_scope('act_prob_'+flag):
        input_crop = tf_observation / 255
        input = tf.transpose(input_crop, [0, 2, 3, 1])  # (?, 80, 80, 4)

        # tf.contrib.layers.conv2d(..., activation_fn=tf.nn.relu,...)
        conv1 = tf.contrib.layers.conv2d(inputs=input, num_outputs=32, kernel_size=8, stride=4)  # (?, 20, 20, 32)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2)  # (?, 10, 10, 64)
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, stride=1)  # (?, 10, 10, 64)

        flat = tf.contrib.layers.flatten(conv3)
        f = tf.contrib.layers.fully_connected(flat, 512)
        all_act_prob = tf.contrib.layers.fully_connected(f, n_actions, activation_fn=tf.nn.softmax)

    with tf.name_scope('loss_'+flag):
        # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
        # this is negative log of chosen action
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act_prob, labels=tf_actions)
        # or in this way:
        # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_actions, self.n_actions), axis=1)
        loss = tf.reduce_mean(neg_log_prob * tf_vt)  # reward guided loss

    with tf.name_scope('train_'+flag):
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)

    return [[tf_observation, tf_actions, tf_vt],
            [all_act_prob, loss, train_op]]


if __name__ == '__main__':
    build_network(0.0001)
