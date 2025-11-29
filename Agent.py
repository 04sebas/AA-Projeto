import random

from numpy import argmax

from AmbienteFarol import Goal, Wall
import numpy as np

class AgenteLearner :
    
    def __init__(self, farol, neural_network):
        
        self.actions = [(-1,0), (1,0), (0,-1), (0,1)] 
        
        self.foundGoal = False
        self.num_steps = 5000
        self.x = farol.agentx
        self.y = farol.agenty
        self.farol = farol
        self.neural_network = neural_network
        self.behavior = set()
        self.path = []
        self.fitness = 0.0
    
    def run_simulation(self, start_x=None, start_y=None):
        
        """Runs the agent's genotype in a fresh environment to get its behavior."""
        
        # --- Reset all state variables ---

        self.foundGoal = False
        self.behavior = set()
        self.path = []
        self.fitness = 0.0

        if start_x is not None and start_y is not None:
            self.x, self.y = start_x, start_y
        else:
            self.x, self.y = self.farol.agentx, self.farol.agenty

        # Add starting position
        self.behavior.add((self.x, self.y))
        self.path.append((self.x, self.y))

        # We need to track the keys *this agent* has for this run

        #for action in self.genotype:
        
        for _ in range(self.num_steps):
            
            
            inputs = np.array([self.x / self.farol.size, self.y / self.farol.size,
                   self.farol.goalx / self.farol.size, self.farol.goaly / self.farol.size])

            output_vector = self.neural_network.forward(inputs)

            action_index = argmax(output_vector)

            reward, found = self.step(action_index)

            self.fitness += reward
            self.behavior.add((self.x, self.y))
            self.path.append((self.x, self.y))

            if found:
                break

    def step(self, action_index):

        action_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = action_map[action_index]
        newx = self.x + dx
        newy = self.y + dy

        if not (0 <= newx < self.farol.size and 0 <= newy < self.farol.size):
            return -1, False

        obj = self.farol.get_object_here(newx, newy)

        if isinstance(obj, Goal):
            self.foundGoal = True
            self.y = newy
            self.x = newx
            return 1000, True

        elif isinstance(obj, Wall):

            return -1, False
        else:
            prev_dist = self.distance_to_goal()
            self.y = newy
            self.x = newx
            new_dist = self.distance_to_goal()
            reward = -0.1
            if new_dist < prev_dist:
                reward += 0.2
            return reward, False

        
    def mutate(self, mutation_rate):
        """Randomly changes some actions in the genotype."""
        self.weights = self.neural_network.weights
        for i in range(len( self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += random.uniform(-0.1, 0.1)
                
        self.neural_network.load_weights(self.weights)

    # --- NEW FITNESS FUNCTION ---
    def getFitness(self):
        return self.fitness
        
    def distance_to_goal(self):
        #return math.sqrt((self.x - self.farol.goalx)**2 + (self.y - self.farol.goaly)**2)
        return abs(self.x - self.farol.goalx) + abs(self.y - self.farol.goaly)

