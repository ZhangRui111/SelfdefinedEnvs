import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import time

from hard_maze import HardMaze
from hard_maze_dqn import DQNBrain
from hard_maze_dqn import store_parameters
from hard_maze_dqn import REPLY_START_SIZE

MAX_EPISODE = 400  # 400
COUNTERS_SIZE = 10
UPDATE_FREQUENCY = 5
WEIGHTS_SAVER_ITER = 10000
LOGS_DATA_PATH = './logs_data/hard_maze/dqn/'
LOGS_EVENTS_PATH = './logs_events/hard_maze/dqn/'


def run_maze(saver, sess, load_step):
    step = 0
    counters_list = []  # store all episodes' count
    rewards_list = []  # store all episodes' reward
    average_counts = []  # store all the average counts on counters

    for episode in range(MAX_EPISODE):
        print('episode:' + str(episode))
        # initial observation
        observation_raw = env.random_reset_hard_maze()
        observation = observation_raw.flatten().reshape([1, RL.n_features])
        final_count = 0  # counter for one episode
        final_reward = 0  # return for one episode

        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation, step)

            # RL take action and get next observation and reward
            if episode % 50 == 0 and episode != 0:
                reward, done, observation_raw_ = env.move(action, True)
            else:
                reward, done, observation_raw_ = env.move(action, False)

            final_reward = final_reward + reward
            observation_ = observation_raw_.flatten().reshape([1, RL.n_features])

            RL.store_transition(observation, action, reward, observation_)
            # start learning on condition 'step > REPLY_START_SIZE', so that we can have
            # enough transition stored for sampling.
            # 'step % UPDATE_FREQUENCY == 0' is for frame-skipping technique.
            if (step > REPLY_START_SIZE) and (step % UPDATE_FREQUENCY == 0):
                RL.learn()
                if step % WEIGHTS_SAVER_ITER == 0:
                    saver.save(sess, LOGS_EVENTS_PATH + '-dqn-' + str(step + load_step))

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1

            # counter for one episode.
            final_count += 1

        print('episode:' + str(episode) + ' | ' + str(final_count))
        counters_list.append(final_count)
        rewards_list.append(final_reward)

        if len(counters_list) >= 20:
            fp = open(LOGS_DATA_PATH + 'counters_record.txt', "a")
            fp.write(str(counters_list))
            fp.close()
            del counters_list[:]
        if len(rewards_list) >= 20:
            fp = open(LOGS_DATA_PATH + 'rewards_record.txt', "a")
            fp.write(str(rewards_list))
            fp.close()
            del rewards_list[:]

        # counter_add(counters, count)  # append the last COUNTERS_SIZE count to counters
        # average the last COUNTERS_SIZE counts and store them in average_counts
        # average_counts.append(np.mean(counters))

    # evaluation(average_counts)

    # end of game
    return counters_list, rewards_list


def counter_add(counters, count):
    if len(counters) >= COUNTERS_SIZE:
        counters.pop(0)  # pop(0) --> FIFO
    counters.append(count)


if __name__ == "__main__":
    # get the maze environment
    env = HardMaze()
    # get the DeepQNetwork Agent
    RL = DQNBrain([4, 2], 3 * 17 * 17,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=2000,
                  memory_size=2000,
                  e_greedy_increment=0.01,
                  output_graph=True,
                  )
    saver, load_step = store_parameters(RL.sess)
    # Calculate running time
    start_time = time.time()

    counters_record, rewards_record = run_maze(saver, RL.sess, load_step)

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    fo = open(LOGS_DATA_PATH + 'running_time_dqn.txt', "w")
    fo.write(str(running_time) + "minutes")
    fo.close()

    fo = open(LOGS_DATA_PATH + 'counters_record.txt', "a")
    fo.write(str(counters_record))
    fo.close()

    fo = open(LOGS_DATA_PATH + 'rewards_record.txt', "a")
    fo.write(str(rewards_record))
    fo.close()
