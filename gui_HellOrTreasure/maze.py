import json
import numpy as np
import time
import tkinter as tk


class Maze(tk.Tk, object):
    def __init__(self, path_to_map, full_observation=True):
        super(Maze, self).__init__()

        map = self._read_map(path_to_map)
        self.height = map['height']
        self.width = map['width']
        self.hells_origin = map['hells']  # all hells' position.
        self.exit_origin = map['exit']  # exit's position.
        self.player_origin = map['player']  # player's position.

        self.unit = 40  # size of one self.unit/pixels.
        self.action_space = ['u', 'd', 'l', 'r']  # four actions: up, down, left, right.
        self.n_actions = len(self.action_space)

        self.title('Gui Maze')
        self.full = full_observation
        self.geometry('{0}x{1}'.format(self.height * self.unit, self.height * self.unit))  # windows background.
        self._build_maze()
        self.observation = self._build_observation()

    def _read_map(self, path_to_map):
        with open(path_to_map) as json_file:
            data = json.load(json_file)
            return data

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=self.height * self.unit, width=self.width * self.unit)  # maze.

        # create grids.
        for c in range(0, self.width * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.height * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.height * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.width * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # coordination of the origin point (center of the top left corner unit).
        self.origin = np.array([self.unit/2, self.unit/2])

        # create hell (black rectangle).
        self.hells_coords = []  # hold all hells' coordination.
        for i in range(len(self.hells_origin)):
            hell_center = self.origin + np.array(
                [self.unit * self.hells_origin[i][0], self.unit * self.hells_origin[i][1]])
            hell = self.canvas.create_rectangle(
                hell_center[0] - 15, hell_center[1] - 15,
                hell_center[0] + 15, hell_center[1] + 15,
                fill='black')
            self.hells_coords.append(self.canvas.coords(hell))

        # create exit (green oval)
        exit_center = self.origin + np.array(
            [self.unit * self.exit_origin[0][0], self.unit * self.exit_origin[0][1]])
        self.exit = self.canvas.create_oval(
            exit_center[0] - 15, exit_center[1] - 15,
            exit_center[0] + 15, exit_center[1] + 15,
            fill='green')

        # create player (yellow oval)
        player_center = self.origin + np.array(
            [self.unit * self.player_origin[0][0], self.unit * self.player_origin[0][1]])
        self.player = self.canvas.create_oval(
            player_center[0] - 15, player_center[1] - 15,
            player_center[0] + 15, player_center[1] + 15,
            fill='yellow')

        # pack all
        self.canvas.pack()

    def _build_observation(self):
        """ Build full observed observation. """
        #  2 -- exit
        # -1 -- hell
        #  1 -- player
        #  0 -- path
        obser = np.zeros((self.width, self.height))
        obser[self.exit_origin[0][0], self.exit_origin[0][1]] = 2
        for item in self.hells_origin:
            obser[item[0], item[1]] = -1

        return obser

    def reset(self):
        self.update()
        self.canvas.delete(self.player)
        # reset the player.
        player_center = self.origin + np.array(
            [self.unit * self.player_origin[0][0], self.unit * self.player_origin[0][1]])
        self.player = self.canvas.create_oval(
            player_center[0] - 15, player_center[1] - 15,
            player_center[0] + 15, player_center[1] + 15,
            fill='yellow')
        # return observation
        if self.full:
            obser = self.observation.copy()
            obser[self.player_origin[0][0], self.player_origin[0][1]] = 1
            return obser.T
        else:
            return self.canvas.coords(self.player)

    def step(self, action):
        s = self.canvas.coords(self.player)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > self.unit:
                base_action[1] -= self.unit
        elif action == 1:  # down
            if s[1] < (self.height - 1) * self.unit:
                base_action[1] += self.unit
        elif action == 2:  # left
            if s[0] > self.unit:
                base_action[0] -= self.unit
        elif action == 3:  # right
            if s[0] < (self.width - 1) * self.unit:
                base_action[0] += self.unit
        else:
            print('illegal action!')

        self.canvas.move(self.player, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.player)  # next state

        # reward function
        if s_ == self.canvas.coords(self.exit):
            reward = 1
            done = True
            info = 'terminal'
        elif s_ in self.hells_coords:
            reward = -1
            done = True
            info = 'terminal'
        else:
            reward = 0
            done = False
            info = 'running'

        if self.full:
            player_real = [int(s_[0] // self.unit), int(s_[1] // self.unit)]
            obser = self.observation.copy()
            obser[player_real[0], player_real[1]] = 1
            return obser.T, reward, done, info
        else:
            return s_, reward, done, info

    def render(self, slt=None):
        if slt is not None:
            time.sleep(slt)
        self.update()


# def my_update():
#     for t in range(10):
#         env.reset()
#         while True:
#             env.render(0.5)
#             a = np.random.random_integers(4)-1
#             print(a)
#             s, r, done, info = env.step(a)
#             if done:
#                 print(info)
#                 break
#
#
# def main():
#     global env
#     env = Maze('./maps/map1.json')
#     env.after(100, my_update)  # Call function update() once after given time/ms.
#     env.mainloop()  # mainloop() to run the application.
#
#
# if __name__ == '__main__':
#     main()
