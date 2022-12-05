# Mohamed Nagy PC, 3 December2022
#################################
#################################

import gym
import numpy as np
import cv2

class QLearningAgent:

    def __init__(self, environment, problem_grid,
                 alpha = 0.02, gamma = 0.99, epsilon = 1.0,
                 epsilon_decay_rate = 0.9995, min_epsilon = .01, seed = 500):

        self.environment    = environment
        self.grid           = problem_grid
        self.state_size     = self.get_state_size(problem_grid)
        self.action_size    = self.environment.action_space.n

        self.alpha                          = alpha     # learning rate
        self.gamma                          = gamma     # discount factor
        self.epsilon = self.initial_epsilon = epsilon   # greedy rate (exploration/exploitation)
        self.epsilon_decay_rate             = epsilon_decay_rate
        self.min_epsilon                    = min_epsilon
        self.seed                           = seed

        self.initialize_Qtable() # initialize Q table of zeros (states x actions)

    def get_state_size(self, grid):
        return tuple([len(dim) + 1 for dim in grid])

    def initialize_Qtable(self):
        self.Qtable = np.zeros(self.state_size + (self.action_size,)) # convert to tuple

    def reset_exploration(self, epsilon = None):
        self.epsilon = self.epsilon if epsilon is not None else self.initial_epsilon

    def reset_episode(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def initial_action_from_Qtable(self, state):
        self.last_state     = discretize(state, self.grid)
        self.last_action    = np.argmax(self.Qtable[tuple(self.last_state)])

        return self.last_action

    def update_Qtable(self, state, reward):
        # Update using SARSA
        self.Qtable[tuple(self.last_state) + (self.last_action,)] += \
            self.alpha *(reward + self.gamma * max(self.Qtable[tuple(state)])
                         - self.Qtable[tuple(self.last_state) + (self.last_action,)])

    def pickAction(self, state, mode = 'train'):
        if mode == 'test':
            action = np.argmax(self.Qtable[tuple(state)])
        else:
            #  0.99
            do_exploration = np.random.uniform(0, 1) < self.epsilon

            if do_exploration:
                action = np.random.randint(0, self.action_size)

            else:
                action = np.argmax(self.Qtable[tuple(state)])


        return action

def run(agent, environment, num_episodes = 20000, mode = 'train', visual = False):
    scores = []

    for episode_idx in range(1, num_episodes + 1):
        # Initialization step
        (state, _)      = environment.reset() # get initial state
        agent.reset_episode()   # decay exploration
        total_reward    = 0
        reached_goal    = False
        action          = agent.initial_action_from_Qtable(state= state)

        while not reached_goal:
            # perform action on the environment, get next state + reward
            state, reward, reached_goal, _, _ = environment.step(action)
            total_reward += reward

            # transform state continuous to discrete
            # TODO: call discretize function to discretize the obtain state at line 83.
            state = discretize(??,??)

            # update Q-table
            if mode == 'train':
                agent.update_Qtable(state= state, reward= reward)
            # pick an action
            action = agent.pickAction(state= state, mode= mode)
            # store last state & action
            agent.last_state, agent.last_action = state, action

            # visualization
            if visual: showUpdate(environment= environment)

        # episode ends! store reward
        scores.append(total_reward)
        print('episode {}/{} ends with reward {}.'.format(episode_idx, num_episodes, total_reward))

        if visual: cv2.destroyAllWindows()
    return scores

def showUpdate(environment, title = 'window'):
    updated_scene = environment.render()
    cv2.imshow(title, updated_scene)
    cv2.waitKey(10)

def run_with_random_actions(environment, timestamp=800, visual= False):
    score = 0

    environment.reset()

    for _ in range(timestamp):
        action = environment.action_space.sample()
        state, reward, terminated, truncated, _ = environment.step(action)
        score += reward
        showUpdate(environment)
        if terminated:
            break

    environment.close()
    if visual: cv2.destroyAllWindows()

    return score

def create_uniform_grid(low, high, bins=(10, 10)):
    #TODO: create and return a grid of the continuous space
    pass

def discretize(sample, grid):
    #TODO: given a state and the grid, return the state location on the grid. (discretization)
    pass

# Try random actions
problem = 'MountainCar-v0'

environment = gym.make(problem, render_mode="rgb_array")

score = run_with_random_actions(environment, visual= True)

print('Final score is {}'.format(score))

grid_size = 10

### Transform the env. from continuous to discrete states.
problem_grid = create_uniform_grid(
    environment.observation_space.low,
    environment.observation_space.high,
    bins = (grid_size, grid_size))

# Initialize an agent
q_agent = QLearningAgent(environment = environment,
                         problem_grid= problem_grid)

# Train
run(agent= q_agent,
    environment= environment,
    num_episodes= 10000,
    mode= 'train',
    visual= False)

# Testing
run(agent= q_agent,
    environment= environment,
    mode= 'test',
    visual= True)
