import numpy as np
import PIL.Image as Image
import random
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt


PROB_RANDOM_RESET = 0.2
# # All kins of special points.
START_POINTS = [[8, 1]]
EXIT_POINTS = [[2, 15], [14, 15]]
DEAD_POINTS = [[8, 5]]
GAMING_POINTS = [[5, 13], [11, 13]]
ROAD_JUMP_POINTS = [[2, 3], [2, 6], [2, 14], [4, 1], [4, 9], [7, 15], [8, 12], [9, 15], [12, 2], [14, 4], [14, 14],
                    [15, 8]]
ROAD_NO_JUMP_POINTS = [[1, 2], [1, 3], [3, 7], [3, 13], [6, 10], [6, 15], [8, 10], [8, 14], [10, 10], [10, 15],
                       [14, 2], [15, 2], [15, 11], [15, 12]]
WALL_POINTS = [[1, 11], [1, 12], [1, 13], [2, 1], [2, 7], [2, 8], [2, 9], [3, 1], [3, 2], [3, 3], [3, 8], [3, 9],
               [3, 10], [3, 11], [3, 12], [4, 6], [4, 12], [4, 13], [4, 14], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],
               [5, 7], [5, 8], [5, 9], [5, 10], [5, 14], [6, 2], [6, 8], [6, 14], [7, 1], [7, 2], [7, 8], [7, 10],
               [7, 11], [7, 14], [9, 1], [9, 2], [9, 8], [9, 10], [9, 11], [9, 14], [10, 2], [10, 8], [10, 14], [11, 2],
               [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 14], [12, 7], [12, 8],
               [12, 12], [12, 13], [12, 14], [13, 1], [13, 2], [13, 4], [13, 5], [13, 10], [13, 11], [13, 12], [13, 13],
               [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [15, 1]]  # Edge wall not included.


class HardMaze(object):
    """ class HardMaze:
        define the HardMaze environment:

        define the action [x, y]:
            x:                   y:
            w -- up              j -- jump
            s -- down            n -- none
            a -- left
            d -- right
        """
    def __init__(self):
        """
        self.reward -- reward for one step.
        self.terminal -- whether one episode ends.
        self.current_p -- agent's current position.
        self.old_p -- agent's last position.
        self.maze -- the maze, which is a 3d array.
        self.old_pixel -- last position's pixel values.
        """
        self.reward = 0
        self.terminal = False
        self.current_p = [8, 1]
        self.old_p = [self.current_p[0], self.current_p[1]]
        self.maze = init_maze(255*np.ones([3, 17, 17]))
        self.old_pixel = get_pixel_from_coord(self.maze, self.current_p)
        self.maze[:, 8, 1] = [255, 0, 0]

    def reset_hard_maze(self):
        """ Reset the HardMaze, put the agent at start point.
        """
        self.reward = 0
        self.terminal = False
        self.current_p = [8, 1]
        self.old_p = [self.current_p[0], self.current_p[1]]
        self.maze = init_maze(255 * np.ones([3, 17, 17]))
        self.old_pixel = get_pixel_from_coord(self.maze, self.current_p)
        self.maze[:, self.current_p[0], self.current_p[1]] = [255, 0, 0]
        return self.maze

    def random_reset_hard_maze(self):
        """ Reset the HardMaze, put the agent at random point, with probability PROB_RANDOM_RESET at start point [8, 1]
        """
        self.reward = 0
        self.terminal = False
        r = random.uniform(0, 1)
        if r < PROB_RANDOM_RESET:
            self.current_p = [8, 1]
        else:
            r = np.random.random_integers(1, 15)
            c = np.random.random_integers(1, 15)
            while [r, c] in WALL_POINTS or [r, c] in DEAD_POINTS:
                r = np.random.random_integers(1, 15)
                c = np.random.random_integers(1, 15)
            self.current_p = [r, c]

        self.old_p = [self.current_p[0], self.current_p[1]]
        self.maze = init_maze(255 * np.ones([3, 17, 17]))
        self.old_pixel = get_pixel_from_coord(self.maze, self.current_p)
        self.maze[:, self.current_p[0], self.current_p[1]] = [255, 0, 0]
        return self.maze

    def move(self, action, show_maze):
        """ Move the agent following action and change self.maze.
        """
        # # Whether watch the training process.
        if show_maze:
            array_to_image(self.maze)
        # # Get new self.current_p
        if action[0] == 'w':
            self.update_current_p(0, -1)
        if action[0] == 's':
            self.update_current_p(0, 1)
        if action[0] == 'a':
            self.update_current_p(1, -1)
        if action[0] == 'd':
            self.update_current_p(1, 1)

        if self.current_p in EXIT_POINTS:  # Exit points
            self.terminal = True
            if self.current_p == [2, 15]:
                self.reward = 52
            else:
                self.reward = 68
        else:
            self.terminal = False
            if self.current_p in DEAD_POINTS:  # Dead points
                self.reward = -40
                self.current_p = [8, 1]
            elif self.current_p in GAMING_POINTS:  # Gaming points
                r = random.uniform(0, 1)
                if r <= 0.2:
                    self.reward = -40
                    self.current_p = [8, 1]
                elif r <= 0.4:
                    self.terminal = True
                    if self.current_p == [2, 15]:
                        self.reward = 52
                    else:
                        self.reward = 68
                elif r <= 0.6:
                    self.reward = -10
                else:
                    self.reward = -1
            elif self.current_p in ROAD_JUMP_POINTS:  # Road jump points
                self.reward = -1
                if action[1] != 'j':
                    self.current_p = self.old_p
            elif self.current_p in ROAD_NO_JUMP_POINTS:  # Road no jump points
                self.reward = -1
                if action[1] == 'j':
                    self.current_p = self.old_p
            elif self.current_p in WALL_POINTS or self.current_p[0] % 16 == 0 or self.current_p[1] % 16 == 0:
                # Wall points
                self.reward = -1
                self.current_p = self.old_p
            else:  # normal road/start point
                self.reward = -1
        # # After moving, re-paint the maze.
        self.maze[:, self.old_p[0], self.old_p[1]] = self.old_pixel  # Recover old position's pixel value.
        self.old_pixel = get_pixel_from_coord(self.maze, self.current_p)
        self.maze[:, self.current_p[0], self.current_p[1]] = [255, 0, 0]

        self.old_p = [self.current_p[0], self.current_p[1]]

        return self.reward, self.terminal, self.maze

    def update_current_p(self, index, incre):
        self.current_p[index] = self.current_p[index] + incre


def init_maze(maze):
    """ Paint the maze.
    """
    for [i, j] in START_POINTS:
        maze[:, i, j] = [255, 165, 40]
    for [i, j] in EXIT_POINTS:
        maze[:, i, j] = [0, 0, 255]
    for [i, j] in DEAD_POINTS:
        maze[:, i, j] = [255, 20, 147]
    for [i, j] in GAMING_POINTS:
        maze[:, i, j] = [0, 128, 0]
    for [i, j] in ROAD_JUMP_POINTS:
        maze[:, i, j] = [255, 255, 0]
    for [i, j] in ROAD_NO_JUMP_POINTS:
        maze[:, i, j] = [220, 220, 220]
    for [i, j] in WALL_POINTS:
        maze[:, i, j] = [40, 40, 40]
    for i in range(17):
        maze[:, 0, i] = [40, 40, 40]
        maze[:, 16, i] = [40, 40, 40]
        maze[:, i, 0] = [40, 40, 40]
        maze[:, i, 16] = [40, 40, 40]
    return maze


def get_pixel_from_coord(maze, point):
    """ Get the pixel values according to coordinate.
    """
    a = point[0]
    b = point[1]
    c = maze[:, a, b][0]
    d = maze[:, a, b][1]
    e = maze[:, a, b][2]
    r = [c, d, e]
    return r


def array_to_image(array):
    """ Convert a 3d array to an image.
    """
    r = Image.fromarray(array[0]).convert('L')
    g = Image.fromarray(array[1]).convert('L')
    b = Image.fromarray(array[2]).convert('L')
    image = Image.merge("RGB", (r, g, b))
    # image.save("./img.png")
    plt.imshow(image)
    plt.pause(0.01)
    plt.ion()
    plt.axis('off')
    plt.show()


def one_sample_road(maze):
    """ Run one_sample_road(), we can see one possible road.
    """
    actions = [['d', 'j'], ['d', 'j'], ['d', 'j'], ['w', 'j'], ['d', 'j'], ['d', 'j'], ['d', 'j'], ['s', 'j'],
               ['d', 'j'], ['d', 'j'], ['d', 'n'], ['d', 'j'], ['d', 'j'], ['d', 'j'], ['d', 'n'], ['d', 'j'],
               ['w', 'j'], ['w', 'n'], ['w', 'j'], ['w', 'j'], ['w', 'j'], ['w', 'j']]
    for a in actions:
        maze.move(a)
        array_to_image(maze.maze)


if __name__ == '__main__':
    maze = HardMaze()
    one_sample_road(maze)
