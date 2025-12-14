import math
import random

from numpy import argmax
from Farol.AmbienteFarol import Goal, Wall, Ground
import numpy as np

class AgenteLearner :
    
    def __init__(self, farol, neural_network):
        self.weights = None
        self.actions =  [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.foundGoal = False
        self.num_steps = 1000
        self.x = farol.agentx
        self.y = farol.agenty
        self.farol = farol
        self.neural_network = neural_network
        self.behavior = set()
        self.path = []
        self.fitness = 0.0

    def surroundings(self):
        observation = self.farol.observation(self.x, self.y, depth=3)
        obs = []

        action_map = [(0, 1),(0, -1),(-1, 0),(1, 0)]  # up, down, right, left

        for i in range(4):  # for each direction
            dx, dy = action_map[i]
            for step in range(1, 4):  # 1 → 3 steps
                obj = observation[i][step - 1]

                if isinstance(obj, Goal):
                    obs.append(0.9)

                elif isinstance(obj, Wall):
                    obs.append(-0.9)

                elif isinstance(obj, Ground):
                    newx = self.x + dx * step
                    newy = self.y + dy * step
                    newDist = self.distance_to_goal(newx, newy)

                    if self.distance_to_goal(self.x, self.y) > newDist:
                        obs.append(0.5)  # moving closer
                    else:
                        obs.append(-0.1)  # moving further

                else:
                    obs.append(-0.9)  # None or outside

        return obs

    def mutate(self, mutation_rate):
        self.weights = self.neural_network.weights
        for i in range(len( self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += random.uniform(-0.1, 0.1)
                
        self.neural_network.load_weights(self.weights)

    def getFitness(self):
        return self.fitness

    def distance_to_goal(self, x, y):
        return (abs(x - self.farol.goalx) + abs(y - self.farol.goaly))/ self.farol.size

    def distance_to_goal_agent(self):
        #return (abs(self.x - self.farol.goalx) + abs(self.y - self.farol.goaly)) / self.farol.size
        dx = self.x - self.farol.goalx
        dy = self.y - self.farol.goaly
        return math.sqrt(dx * dx + dy * dy) / self.farol.size

    def step(self, action_index):

        action_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = action_map[action_index]
        newx, newy = self.x + dx, self.y + dy

        if not (0 <= newx < self.farol.size and 0 <= newy < self.farol.size):
            return -1, False

        obj = self.farol.get_object_here(newx, newy)

        if isinstance(obj, Goal):
            self.foundGoal = True
            self.y = newy
            self.x = newx
            return 1500, True

        elif isinstance(obj, Wall):

            return -1, False
        else:
            prev_dist = self.distance_to_goal(self.x, self.y)
            self.y = newy
            self.x = newx

            new_dist = self.distance_to_goal(self.x, self.y)
            reward = -0.05
            if new_dist < prev_dist:
                reward += 1.0
            return reward, False


    def run_genetic_simulation(self, start_x=None, start_y=None):
        self.foundGoal = False
        self.behavior = set()
        self.path = []
        self.fitness = 0.0

        if start_x is not None and start_y is not None:
            self.x, self.y = start_x, start_y
        else:
            self.x, self.y = self.farol.agentx, self.farol.agenty

        self.behavior.add((self.x, self.y))
        self.path.append((self.x, self.y))

        for _ in range(self.num_steps):

            obs = self.surroundings()

            inputs = np.array([
                self.x / self.farol.size,
                self.y / self.farol.size,
                *obs,
                self.distance_to_goal(self.x, self.y)
            ])

            output_vector = self.neural_network.forward(inputs)

            action_index = argmax(output_vector)

            reward, found = self.step(action_index)

            self.fitness += reward
            self.behavior.add((self.x, self.y))
            self.path.append((self.x, self.y))

            if found:
                break

    def run_DQN_simulation(self, start_x=None, start_y=None, dqnSim=None):

        self.num_steps = 500
        dqnSim.episode_reward = 0
        dqnSim.done= False
        stepsDone = 0
        episodeReward = 0
        self.path = []

        if start_x is not None and start_y is not None:
            self.x, self.y = start_x, start_y
        else:
            self.x, self.y = self.farol.agentx, self.farol.agenty

        for _ in range(self.num_steps):

            state = dqnSim.getState(self)
            action_index = dqnSim.select_action(state, dqnSim.epsilon)

            reward, done = self.step(action_index)
            self.path.append((self.x, self.y))

            episodeReward += reward

            next_state = dqnSim.getState(self)

            # Store transition in memory
            dqnSim.memory.append((state, action_index, reward, next_state, done))

            # Optimize model
            dqnSim.optimize_model()

            stepsDone += 1

            if done:
                print(f"Goal reached in {stepsDone} steps!")
                return episodeReward, self.path

        return episodeReward, self.path