"""
This is a simulation environment for FIFA 18's shooting.
"""

import numpy as np
import time

ENV_H = 9
ENV_W = 9


class SingleMiniShooting:
    def __init__(self):
        self.action_space = ['wh', 'wl', 'wn', 'ah', 'al', 'an', 'sh', 'sl', 'sn', 'dh', 'dl', 'dn', 'nn']
        # self.action_space = ['0',   '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12']
        self.n_actions = len(self.action_space)
        self.n_feature_h = ENV_H
        self.n_feature_w = ENV_W
        self.location = [4, 2]
        self.good_point = [(3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6)]
        self.goal = [(3, 7), (3, 8), (4, 7), (4, 8), (5, 7), (5, 8)]
        self.env = self._build_env()

        self.goal_reward = 4
        self.miss_reward = -4
        self.cross_line_reward = -4
        self.move_reward = 0
        self.still_reward = -1

    def _build_env(self):
        env = np.zeros((self.n_feature_h, self.n_feature_w))
        for position in self.good_point:
            env[position] = 2  # "2" represents points where shooting can get points.
        for position in self.goal:
            env[position] = 1  # "1" represents the goal's points.
        env[(self.location[0], self.location[1])] = 9  # "9" represent the player's position.
        return env

    def reset(self):
        # print('----------reset----------')
        self.location = [4, 2]
        self.update_env()

        return self.env

    def step(self, action):
        """
        take action and change the env.
          every move (even still) with reward -1;
          shoot but miss the goal with reward -2;
          cross over the border line with reward -2;
          enter the goal's position with reward -2;
          shoot and score with reward 2;
        :param action: integer in [0, 12].
        :return:
        """
        # Group all actions.
        shoot_high = [0, 3, 6, 9]
        shoot_low = [1, 4, 7, 10]
        shoot_null = [2, 5, 8, 11, 12]
        action_w = [0, 1, 2]
        action_a = [3, 4, 5]
        action_s = [6, 7, 8]
        action_d = [9, 10, 11]

        # reward, terminal, state
        terminal = False
        reward = self.move_reward

        if action in shoot_high:
            # shoot high can score in self.good_point with action 'w' or 's' while shooting.
            terminal = True
            current_pos = (self.location[0], self.location[1])
            if (action in action_w or action in action_s) and current_pos in self.good_point:
                print('Congratulations, SCORE!')
                reward = self.goal_reward
            else:
                reward = self.miss_reward
        elif action in shoot_low:
            # shoot low cannot score at anytime, anywhere.
            terminal = True
            reward = self.miss_reward
        else:
            # shoot_null
            if action in action_w:
                # move up
                if self.location[0] == 0:
                    # Cross border
                    print('Warning: cross border.')
                    terminal = True
                    reward = self.cross_line_reward
                else:
                    self.location[0] -= 1
            elif action in action_s:
                # move down
                if self.location[0] == self.n_feature_h - 1:
                    # Cross border
                    print('Warning: cross border.')
                    terminal = True
                    reward = self.cross_line_reward
                else:
                    self.location[0] += 1
            elif action in action_a:
                # move left
                if self.location[1] == 0:
                    # Cross border
                    print('Warning: cross border.')
                    terminal = True
                    reward = self.cross_line_reward
                else:
                    self.location[1] -= 1
            elif action in action_d:
                # move right
                if self.location[1] == self.n_feature_w - 1:
                    # Cross border
                    print('Warning: cross border.')
                    terminal = True
                    reward = self.cross_line_reward
                else:
                    self.location[1] += 1
            else:
                # keep still
                reward = self.still_reward

        current_pos = (self.location[0], self.location[1])
        if current_pos in self.goal:
            # enter the goal's position.
            print('Warning: enter the goal.')
            terminal = True
            reward = self.cross_line_reward

        self.update_env()

        if terminal is True:
            self.reset()

        return self.env, reward, terminal

    def update_env(self):
        self.env = self._build_env()

    def show_env(self):
        # update current location
        # self._build_env()
        # self.env[self.location[0], self.location[1]] = 9  # "9" represents player.
        # show the env
        print(self.env)
        time.sleep(0.1)


def main():
    # get the env environment
    env = SingleMiniShooting()
    # env.show_env()
    print(env.action_space)

    # ['wh', 'wl', 'wn', 'ah', 'al', 'an', 'sh', 'sl', 'sn', 'dh', 'dl', 'dn', 'nn']
    # ['0',   '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12']

    # test actions
    test_action = [11, 11, 11, 0, 2, 2, 2, 2, 2, 2, 11, 11, 11, 11, 6, 5, 5, 5, 5, 5]
    for i in range(len(test_action)):
        # action = [np.random.random_integers(0, 5), np.random.random_integers(0, 3)]
        action = test_action[i]
        # print(action)
        e, r, t = env.step(action)
        env.show_env()
        print(r, ' | ', t)


if __name__ == "__main__":
    main()
