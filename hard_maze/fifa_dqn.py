import numpy as np
import os
import tensorflow as tf
import time

from brain import DeepQNetwork

# [[w, a, s, d], [None, space], [None, ctrl, i]] -- 1 means selected action while others are 0.
N_ACTIONS = [[1, 0, 0, 0], [1, 0], [1, 0, 0]]
N_FEATURES = 128  # length of input image.
MAX_EPISODE = 500
REPLY_START_THRESHOLD = 1000  # int: when to start experience reply.
UPDATE_FREQUENCY = 1  # interval for frame-skipping technique.

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run():
    """Algorithm: Deep Q Network
    """
    count_all = 0  # counter for all episode.
    scores = []  # store the score

    for episode in range(MAX_EPISODE):
        print('episode:' + str(episode))
        # initial observation
        # ToDo(FIFA): API 1 -- reset_fifa()
        observation = env.reset_fifa()  # observation: such as [225, 225, 0, 0, ... , 225, 225]
        # counter for one episode
        count_one = 0  # counter for one episode.

        while True:
            # RL choose action based on observation.
            action = RL.choose_action(observation, count_all)
            # RL take action and get next observation and reward.
            # ToDo(FIFA): API 2 -- update(action)
            reward, done, observation_ = env.update(action)
            # store transition in memory pool.
            RL.store_transition(observation, action, reward, observation_)
            # start learning on condition 'step > REPLY_START_SIZE', so that we can have
            # enough transition stored for sampling.
            # 'step % UPDATE_FREQUENCY == 0' is for frame-skipping technique.
            if (count_all > REPLY_START_THRESHOLD) and (count_all % UPDATE_FREQUENCY == 0):
                RL.learn()

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break

            count_all += 1
            count_one += 1

        scores.append(reward)  # If one episode is down, then store the score.

    evaluation()


def evaluation():
    """Evaluate the performance of AI.
    """
    pass


if __name__ == '__main__':
    # get the DeepQNetwork Agent
    RL = DeepQNetwork(N_ACTIONS, N_FEATURES, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9,
                      replace_target_iter=300, memory_size=2000, e_greedy_increment=None,
                      e_policy_threshold=REPLY_START_THRESHOLD,
                      )

    # Calculate running time
    start_time = time.time()
    run()
    end_time = time.time()
    running_time = (end_time - start_time) / 60
    print('running_time: ' + str(running_time) + 'min')
