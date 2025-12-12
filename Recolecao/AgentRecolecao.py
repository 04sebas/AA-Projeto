import random

from numpy import argmax

import numpy as np

from Recolecao.Entities import Delivery, Resource, Ground, Wall


class AgenteRecolecao:

    def __init__(self, ambiente, neural_network):

        self.y = None
        self.x = None
        self.yInitial = None
        self.xInitial = None
        self.weights = None
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.num_steps = 250
        self.amb = ambiente
        self.neural_network = neural_network
        self.behavior = set()
        self.path = []
        self.fitness = 0.0

        self.resources = set()
        self.delivery = set()

        self.random = False
        self.carrying = False
        self.currentGoal = None
        self.delivered = 0
        self.done = False
        self.maxCollected = 5
    def surroundings(self):
        observation = self.amb.observation(self.x, self.y, depth=3)
        obs = []

        for i in range(4):  # for each direction
            for step in range(1, 4):  # 1 → 3 steps
                obj = observation[i][step - 1]

                if isinstance(obj, Delivery):
                    if self.carrying:
                        obs.append(1.0)
                    else:
                        obs.append(0.3)
                elif isinstance(obj, Resource):
                    if not self.carrying:
                        obs.append(1.0)
                    else:
                        obs.append(0.2)

                elif isinstance(obj, Wall):
                    obs.append(-0.9)

                elif isinstance(obj, Ground):
                    obs.append(0.1)

                else:
                    obs.append(-0.9)  # None or outside

        return obs


    def setPosition(self, x, y):
        self.x = x
        self.y = y
        self.yInitial = y
        self.xInitial = x

    def setAmbiente(self, ambiente):
        self.amb = ambiente

    def mutate(self, mutation_rate):
        self.weights = self.neural_network.weights
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += random.uniform(-0.1, 0.1)

        self.neural_network.load_weights(self.weights)

    def getFitness(self):
        total_fitness = self.fitness + (self.delivered * 10)
        return total_fitness

    def step(self, action_index):

        action_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = action_map[action_index]
        newx, newy = self.x + dx, self.y + dy

        if not (0 <= newx < self.amb.size and 0 <= newy < self.amb.size):
            return -1

        obj = self.amb.get_object_here(newx, newy)

        if isinstance(obj, Resource):
            if not self.carrying: # Não leva nada - > carrega o resource e começa a procurar um deliveryPoint
                self.y = newy
                self.x = newx
                self.amb.consume(obj)
                self.carrying = True
                self.resources.discard(obj)
                self.random = False
                if self.delivery:
                    self.currentGoal = random.choice(list(self.delivery))
                return 5
            else:                    # Encontrou um resource mas já carrega um, guarda o resource na sua memoria
                self.resources.add(obj)
                return 1

        elif isinstance(obj, Wall):
            return -1

        elif isinstance(obj, Delivery):
            self.delivery.add(obj)

            if self.carrying:
                self.carrying = False
                self.random = False
                self.delivered += 1
                if self.resources:
                    self.currentGoal = random.choice(list(self.resources))
                return 10
            return 1

        else:
            self.y = newy
            self.x = newx
            return 0.1

    def comunicar(self, agente): # Agentes trocam informaçoes e
        pass
        self.delivery.update(agente.delivery)
        self.resources.update(agente.resources)

        agente.resources = self.resources
        agente.delivery = self.delivery

        total_delivered = self.delivered + agente.delivered
        self.delivered = total_delivered
        agente.delivered = total_delivered

        if self.delivered >= self.maxCollected:
            self.done = True
        if agente.delivered >= agente.maxCollected:
            agente.done = True

    def run_genetic_simulation(self):

        self.reset()

        self.behavior.add((self.x, self.y))
        self.path.append((self.x, self.y))

        for _ in range(self.num_steps):
            if self.done:
                break

            obs = self.surroundings()

            if self.currentGoal:
                goal_x = self.currentGoal.x / self.amb.size
                goal_y = self.currentGoal.y / self.amb.size
            else:
                goal_x = goal_y = 0.0

            inputs = np.array([
                self.x / self.amb.size,
                self.y / self.amb.size,
                *obs,
                goal_x,
                goal_y,
                float(self.carrying)  # Add carrying state
            ])

            output_vector = self.neural_network.forward(inputs)
            action_index = argmax(output_vector)
            reward = self.step(action_index)


            self.fitness += reward

            self.behavior.add((self.x, self.y))
            self.path.append((self.x, self.y))

    def reset(self):
        self.behavior = set()
        self.path = []
        self.fitness = 0.0
        self.carrying = False
        self.delivered = 0
        self.delivery = set()
        self.resources = set()
        self.done = False
        self.currentGoal = None
        self.random = False
        self.x = self.xInitial
        self.y = self.yInitial