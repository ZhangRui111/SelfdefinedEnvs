# maze_dqn
## Introduction to the environment
Applying deep Q network(dqn) to a two-dimensions maze environment such as following:
```markdown
  6  0  0  0  0
  0  0  0  0  0
  0  0 -1  0  0
  0  0  0  0  0
 -1  0  0  0  1
```
or
```markdown
  0  0  0  0  0
  0  0  6  0  0
  0  0 -1  0  0
  0  0  0  0  0
 -1  0  0  0  1
```
`6` means agent's current location

`0` means available path with reward 0

`-1` means 'hole' with reward -1 and game will fail

`1` means 'exit' with reward 1 and game will success

## DQN Algorithm
![image](https://github.com/ZhangRui111/myMaze/tree/master/maze_dqn/images/dqn.png)

## Model graph
![image](https://github.com/ZhangRui111/myMaze/tree/master/maze_dqn/images/dqn_graph.png)

## About codes file
1. `maze.py` : Offer a maze environment.
2. `run_this.py` : Run and evaluate the DQN.
3. `brain.py` : Q learning brain, which is a brain of the agent.

## How to run
`python run_this.py`

Jus for reference: 

It takes me about 3 hours to run the full program(800 episodes) in one GTX1080Ti GPUs. 

But in fact, after 200 episodes, the result is quite well.

## Codes
### maze.py
```markdown
class Maze:
    def __init__(self):...

    def _build_maze(self):...

    def reset_maze(self):...

    def update(self, action):...

    def show_maze(self):...

    def get_bad_point(self):...

    def get_good_point(self):

```
`def _build_maze(self)` : Build a new maze and store the maze into self.maze.

`def reset_maze(self)` : Reset the self.maze and put agent in start point.

`def update(self, action)` : The maze accept an action and return the reward, terminal(whether the game is terminated), next_observation.

`def show_maze(self)` : Show the current maze.

### run_this.py
```markdown
MAX_EPISODE = 800
REPLY_START_SIZE = 1000
COUNTERS_SIZE = 10
UPDATE_FREQUENCY = 5
```
```markdown
def run_maze():...

def evaluation(average_counts):...

def counter_add(counters, count):...  # called by the evaluation function.

def visualize_q_value(actions, width, height, bad_point, good_point):...  # called by the evaluation function.

def print_arrow(action):...  # called by the evaluation function.

def phi(observation):...
```
`def run_maze():` Start the learning and this function is **the key algorithm**.

`def evaluation(average_counts):` Evaluate the learning result.

`def phi(observation):` funtion φ : Used for image preprocessing. Here it is a empty function.
```markdown
if __name__ == "__main__":
    # get the maze environment
    env = Maze()
    # get the DeepQNetwork Agent
    RL = DeepQNetwork(..)
    run_maze()
```


### brain.py
```markdown
class DeepQNetwork:
    def __init__(self):...
    def _build_net(self):...
    def store_transition(self, s, a, r, s_):...
    def choose_action(self, observation, step):...
    def choose_action_eval(self, observation):...
    def learn(self):...
```
`def __init__(self):` Initialize some paremeters and define target net's parameters updating operation. Also start session here.

`def _build_net(self):` Build evaluate_net and target_net. The target_net is untrainable while the evaluate_net's weights are updating at every step.

`def store_transition(self, s, a, r, s_):` Store one transition(s, a, r, s_) at one time.

`def choose_action(self, observation, step):` When `step < REPLY_START_SIZE:`, utterly choose an action randomly in order to make sure that the experience replay buffer pool could have as many as different episodes, both successful episodes(necessary for training) and failed episodes; When `step >= REPLY_START_SIZE:`, choose an action based on epsilon-greedy policy.

`def choose_action_eval(self, observation):` Utterly choose an action based on greedy policy, so that we can visualize the Q value for every state in maze.

`def learn(self):` Check to replace target parameters, sample batch memory from all memory, train, increase epsilon.

## About Evaluation
1. The maze is 5x5 in size. From the starting point to destination, it takes agent at least 8 steps. In the running result graph above, the x axis is episode from 1 to 300. In one episode, the agent start from the starting point and get a reward -1(failed) or 1(successful). If the agent failed to reach the destination, the steps for this episode will be set as a big number such as 100 steps. If the agent successfully get to the destination with a reward 1, then the real number of steps will be recorded. I have one list for ten elements. The last ten episodes’ steps will be recorded here for average operation. So the y axis in the graph is the last ten episodes’ average steps, which is used for evaluation.
2. We can see that the steps to destination is very large (about 100 steps) at the very beginning. At that time, the agent is in the exploration stage and storing every transition for experience reply. Then it start learning or training. In the following episodes, the steps to destination is decreasing gradually. After 200 episodes, the steps stabilize at 8 or 9, indicating the agent has learned how to get to the destination while avoiding the holes. But as a result of epsilon-greedy policy, it still has chance to spend more steps or even failed to get to the destination.
3. About the peak of wave reaching 160: At one episode, the agent takes much more steps to get to the destination successfully, while the failed steps is set to 100 steps. So we have a very high wave crest at the beginning.
