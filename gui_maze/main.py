import numpy as np
from gui_maze.maze import Maze


def my_update():
    for t in range(10):
        steps = 0
        env.reset()
        while True:
            env.render(0)
            a = np.random.random_integers(4)-1
            s, r, done, info = env.step(a)
            steps += 1
            # print('action:{0} | reward:{1} | done: {2}'.format(a, r, done))
            if done:
                print('{0} -- {1}'.format(info, steps))
                env.render(0)
                break


def main():
    global env
    env = Maze('./maps/map2.json')
    env.after(100, my_update)  # Call function update() once after given time/ms.
    env.mainloop()  # mainloop() to run the application.


if __name__ == '__main__':
    main()
