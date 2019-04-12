"""
This is a maze environment for deep Q network.
"""

import numpy as np
import time

MAZE_W = 5
MAZE_H = 5
# MAZE_H = 3
# MAZE_W = 3


class Maze:
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = MAZE_H*MAZE_W
        self.location = [0, 0]
        self._build_maze()

    def _build_maze(self):
        self.maze = np.zeros((MAZE_W, MAZE_H))
        # start point.
        self.start_point = (0, 0)
        self.maze[self.start_point] = 0
        # end point with reward 1.
        self.good_point = (MAZE_W-1, MAZE_H-1)
        self.maze[self.good_point] = 1
        # end point with reward -1.
        # self.bad_point = [((MAZE_W - 1) // 2, (MAZE_H - 1) // 2), (0, MAZE_H - 1), (MAZE_W - 1, 0)]
        self.bad_point = [((MAZE_W - 1) // 2, (MAZE_H - 1) // 2), (MAZE_W - 1, 0)]
        # self.bad_point = [(0, MAZE_H - 1)]
        for i in self.bad_point:
            self.maze[i] = -1

    def reset_maze(self):
        # print('----------reset----------')
        self.location = [0, 0]
        # self.show_maze()

        return self.maze.reshape(1, self.n_features)

    def update(self, action):
        # action: up
        if action == 0:
            if self.location[0] == 0:
                pass
            else:
                self.location[0] -= 1
        # action: down
        if action == 1:
            if self.location[0] == MAZE_W-1:
                pass
            else:
                self.location[0] += 1
        # action: left
        if action == 2:
            if self.location[1] == 0:
                pass
            else:
                self.location[1] -= 1
        # action: right
        if action == 3:
            if self.location[1] == MAZE_H-1:
                pass
            else:
                self.location[1] += 1
        # reward, terminal, state
        terminal = False
        reward = 0
        if (self.location[0], self.location[1]) in self.bad_point:
            terminal = True
            reward = -1
        if (self.location[0], self.location[1]) == self.good_point:
            terminal = True
            reward = 1
        # when terminal
        if terminal:
            if reward == 1:
                # good ending
                self.show_maze()
            # bad ending
            self.reset_maze()
        else:
            self.show_maze()

        return reward, terminal, self.maze.reshape(1, self.n_features)

    def show_maze(self):
        # update current location
        self._build_maze()
        self.maze[self.location[0], self.location[1]] = 6
        # show the maze
        print(self.maze)
        # time.sleep(0.1)

    def get_bad_point(self):
        return self.bad_point

    def get_good_point(self):
        return self.good_point


if __name__ == "__main__":
    # get the maze environment
    env = Maze()
    print(env.action_space)
