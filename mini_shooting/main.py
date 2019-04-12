import numpy as np
import os
import tensorflow as tf
import time

from shared.utils import restore_parameters, save_parameters, write_ndarray, read_ndarray, write_file
from shared.single_action_env import SingleMiniShooting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train_model(brain, if_REINFORCE=False, if_a2c=False):
    env = SingleMiniShooting()

    if if_REINFORCE is True or if_a2c is True:
        log_data_list = np.arange(5).reshape((1, 5))
    else:
        log_data_list = np.arange(6).reshape((1, 6))

    saver, load_episode = restore_parameters(brain.sess, brain.graph_path)

    # train for numbers of episodes
    total_steps = 0
    probs_list = []
    write_file(brain.graph_path + 'probs', 'probs_list\n', True)

    for i_episode in range(brain.max_episode):

        observation = env.reset()
        state = np.stack([observation] * 4)

        ep_reward = 0
        num_step = 0
        start_time = time.time()

        while True:
            # time.sleep(0.1)
            if if_a2c is True:
                action, probs = brain.actor.choose_action(state)
            elif if_REINFORCE is True:
                action, probs = brain.choose_action(state)
            else:
                action = brain.choose_action(state)

            # print('action: {}'.format(action))
            observation_, reward, done = env.step(action)
            if reward == 2:
                print('------------------------- SCORE ---------------------------')
            next_state = np.concatenate([state[1:], np.expand_dims(observation_, 0)], axis=0)

            brain.store_transition(state, action, reward, next_state)

            ep_reward += reward
            num_step += 1
            state = next_state
            total_steps += 1

            if if_REINFORCE is False and i_episode > brain.replay_start:
                brain.learn(done)

            if done:

                if if_REINFORCE is True:
                    brain.learn()
                # Logs issues, i.e., save the log info.
                record_num = 6

                if if_REINFORCE is True or if_a2c is True:
                    record_num = 5

                    print('episode: ', i_episode, ' | reward: ', ep_reward, 'probs', probs,
                          'episode_time', time.time() - start_time)

                    probs_list.append(probs)
                    if len(probs_list) % brain.hp.OUTPUT_SAVER_ITER == 0:
                        write_file(brain.graph_path + 'probs', probs_list, False)
                        probs_list = []

                    new_data = np.array([i_episode, ep_reward, num_step, total_steps,
                                         time.time() - start_time]).reshape((1, record_num))
                    log_data_list = np.concatenate((log_data_list, new_data))

                if if_REINFORCE is False and if_a2c is False:
                    # record_num = 6
                    print('episode: ', i_episode, ' | reward: ', ep_reward, 'epsilon', brain.epsilon,
                          'episode_time', time.time() - start_time)

                    new_data = np.array([i_episode, ep_reward, num_step, total_steps, brain.epsilon,
                                         time.time() - start_time]).reshape((1, record_num))
                    log_data_list = np.concatenate((log_data_list, new_data))

                if log_data_list.shape[0] % brain.hp.OUTPUT_SAVER_ITER == 0:
                    write_ndarray(brain.graph_path + 'data', np.array(log_data_list))
                    log_data_list = log_data_list[-1, :].reshape(1, record_num)
                if i_episode % brain.hp.WEIGHTS_SAVER_ITER == 0 and i_episode != 0:
                    save_parameters(brain.sess, brain.graph_path, saver,
                                    brain.graph_path + '-' + str(load_episode + i_episode))
                break


def main():
    # #parameters adjusting.
    # for learning_rate in [1E-4, 1E-3, 1E-5]:
    #     for discount_factor in [0.99, 0.5]:
    #         tf.reset_default_graph()
    #         token = str(learning_rate) + str(discount_factor)
    #         train_model(token, learning_rate, discount_factor)

    tf.reset_default_graph()
    # #choose model
    model = 'dqn_2015'

    if model == 'double_dqn':
        print('double_dqn')
        from brain.double_dqn import DeepQNetwork
        from network.network_double_dqn import build_network
        from hyper_paras.hp_double_dqn import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'double_dqn'  # token is useful when para-adjusting (tick different folder)
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)

    elif model == 'dueling_dqn':
        print('dueling_dqn')
        from brain.dueling_dqn import DeepQNetwork
        from network.network_dueling_dqn import build_network
        from hyper_paras.hp_dueling_dqn import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'dueling_dqn'  # token is useful when para-adjusting (tick different folder)
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)
    elif model == 'pri_dqn':
        print('pri_dqn')
        from brain.pri_dqn import DeepQNetwork
        from network.network_pri_dqn import build_network
        from hyper_paras.hp_pri_dqn import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'pri_dqn'  # token is useful when para-adjusting (tick different folder)
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)
    elif model == 'REINFORCE':
        print('REINFORCE')
        from brain.REINFORCE import REINFORCE
        from network.network_REINFORCE import build_network
        from hyper_paras.hp_REINFORCE import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'REINFORCE'  # token is useful when para-adjusting (tick different folder)
        brain = REINFORCE(hp=hp, token=token, network_build=bn)
        train_model(brain, if_REINFORCE=True)
    elif model == 'a2c_v':
        print('a2c_v')
        from brain.a2c_v import A2C
        from network.network_a2c_v import build_actor_network, build_critic_network
        from hyper_paras.hp_a2c_v import Hyperparameters

        hp = Hyperparameters()
        actor_bn = build_actor_network()
        critic_bn = build_critic_network()
        token = 'a2c_v'  # token is useful when para-adjusting
        brain = A2C(hp=hp, token=token, network_actor=actor_bn, network_critic=critic_bn)
        train_model(brain, if_a2c=True)
    elif model == 'a2c_q':
        print('a2c_q')
        from brain.a2c import A2C
        from network.network_a2c import build_actor_network, build_critic_network
        from hyper_paras.hp_a2c import Hyperparameters

        hp = Hyperparameters()
        actor_bn = build_actor_network()
        critic_bn = build_critic_network()
        token = 'a2c_q'  # token is useful when para-adjusting
        brain = A2C(hp=hp, token=token, network_actor=actor_bn, network_critic=critic_bn)
        train_model(brain, if_a2c=True)
    else:
        print('No model satisfied, try dqn_2015!')
        from brain.dqn_2015 import DeepQNetwork
        from network.network_dqn_2015 import build_network
        from hyper_paras.hp_dqn_2015 import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'dqn_2015'  # token is useful when para-adjusting
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)


if __name__ == '__main__':
    main()
