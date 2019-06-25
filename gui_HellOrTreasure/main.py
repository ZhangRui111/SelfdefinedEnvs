import numpy as np
from gui_HellOrTreasure.maze import Maze


def my_update():
    for t in range(10):
        s = env.reset()
        print(s)
        while True:
            env.render(0.1)
            a = np.random.random_integers(4)-1
            s, r, done, info = env.step(a)
            print('action:{0} | reward:{1} | done: {2}'.format(a, r, done))
            print(s)
            print('\n')
            if done:
                print(info)
                print("--------------------------------------")
                env.render(0.1)
                break

    # end of game
    print('game over')
    env.destroy()


def main():
    global env
    env = Maze('./maps/map1.json', full_observation=False)
    env.after(100, my_update)  # Call function update() once after given time/ms.
    env.mainloop()  # mainloop() to run the application.


if __name__ == '__main__':
    main()
