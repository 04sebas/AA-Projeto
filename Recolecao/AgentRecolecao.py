import random

from numpy import argmax

import numpy as np

from Recolecao.Entities import Delivery, Resource, Ground, Wall


class AgenteRecolecao:

    def __init__(self, ambiente, neural_network):

        self.randomStepNum = 0
        self.y = None
        self.x = None
        self.yInitial = None
        self.xInitial = None
        self.weights = None
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.num_steps = 1500
        self.amb = ambiente
        self.neural_network = neural_network
        self.behavior = set()
        self.path = []
        self.fitness = 0.0
        self.goalStepsWithoutProgress = 0
        self.lastGoalDistance = None
        self.maxGoalSteps = 75
        self.resources = set()
        self.delivery = set()

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
                    self.delivery.add(obj)
                    if self.carrying:
                        if self.currentGoal is None:
                            self.currentGoal = obj
                        obs.append(0.9)
                    else:
                        obs.append(0.1)

                elif isinstance(obj, Resource):
                    self.resources.add(obj)
                    if self.carrying:
                        obs.append(0.1)
                    else:
                        if self.currentGoal is None:
                            self.currentGoal = obj
                        obs.append(0.9)

                elif isinstance(obj, Wall):
                    obs.append(-0.9)

                elif isinstance(obj, Ground):
                    obs.append(0.1)

                else:
                    obs.append(-0.9)  # None is outside

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
        total_fitness = self.fitness
        return total_fitness

    def step(self, action_index):

        action_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = action_map[action_index]
        newx, newy = self.x + dx, self.y + dy

        if not (0 <= newx < self.amb.size and 0 <= newy < self.amb.size):
            return -1

        obj = self.amb.get_object_here(newx, newy)

        if isinstance(obj, Resource):
            self.y = newy
            self.x = newx
            if not self.carrying: # Não leva nada - > carrega o resource
                self.amb.consume(obj)
                self.carrying = True
                self.currentGoal = None
                self.goalStepsWithoutProgress = 0
                self.lastGoalDistance = None
                self.resources.discard(obj)
                return 100
            else:
                return 1

        elif isinstance(obj, Wall):
            return -1

        elif isinstance(obj, Delivery):
            self.y = newy
            self.x = newx
            if self.carrying:
                self.carrying = False
                self.currentGoal = None
                self.goalStepsWithoutProgress = 0
                self.lastGoalDistance = None
                self.delivered += 1
                return 200
            else:
                return 1
        else:
            self.y = newy
            self.x = newx
            return 0.1

    def run_genetic_simulation(self):

        self.reset()

        self.behavior.add((self.x, self.y))
        self.path.append((self.x, self.y))


        for _ in range(self.num_steps):

            obs = self.surroundings()

            if self.currentGoal is None:
                if not self.carrying:
                    if self.resources:
                        goals = list(self.resources)
                        self.currentGoal = self.choose_closest(goals)

                    else:
                        self.randomStepNum += 1

                        stepReward = self.randomStep()
                        self.fitness += stepReward

                        self.behavior.add((self.x, self.y))
                        self.path.append((self.x, self.y))
                        continue
                else:
                    if self.delivery:
                        goals = list(self.delivery)
                        self.currentGoal = self.choose_closest(goals)
                    else:
                        self.randomStepNum += 1
                        stepReward = self.randomStep()
                        self.fitness += stepReward

                        self.behavior.add((self.x, self.y))
                        self.path.append((self.x, self.y))
                        continue

            current_distance = abs(self.x - self.currentGoal.x) + abs(self.y - self.currentGoal.y)

            if self.lastGoalDistance is None or current_distance < self.lastGoalDistance:
                self.lastGoalDistance = current_distance
                self.goalStepsWithoutProgress = 0
            else:
                self.goalStepsWithoutProgress += 1

            if self.goalStepsWithoutProgress >= self.maxGoalSteps:
                self.currentGoal = None
                self.goalStepsWithoutProgress = 0
                self.lastGoalDistance = None
                continue

            goalx, goaly = self.currentGoal.x / self.amb.size, self.currentGoal.y / self.amb.size
            inputs = np.array([
                self.x / self.amb.size,
                self.y / self.amb.size,
                *obs, goalx,goaly,])

            output_vector = self.neural_network.forward(inputs)
            action_index = argmax(output_vector)
            stepReward = self.step(action_index)

            self.fitness += stepReward

            self.behavior.add((self.x, self.y))
            self.path.append((self.x, self.y))

    def run_DQN_simulation(self, start_x=None, start_y=None, dqnSim=None):

        self.num_steps = 1500
        dqnSim.episode_reward = 0
        dqnSim.done= False
        stepsDone = 0
        episodeReward = 0
        self.path = []

        if start_x is not None and start_y is not None:
            self.x, self.y = start_x, start_y
        else:
            self.x, self.y =  self.amb.random_valid_position()

        for _ in range(self.num_steps):

            self.surroundings() # Observa antes de random step, caso haja algum goal
            state = dqnSim.getState(self)
            reward = 0
            action_index = 0

            if self.currentGoal is None:
                if not self.carrying:
                    if self.resources:
                        goals = list(self.resources)
                        self.currentGoal = self.choose_closest(goals)

                    else:
                        self.randomStepNum += 1
                        stepReward, action_index = self.randomStep()
                        reward += stepReward
                        self.path.append((self.x, self.y))
                else:
                    if self.delivery:
                        goals = list(self.delivery)
                        self.currentGoal = self.choose_closest(goals)

                    else:
                        self.randomStepNum += 1
                        stepReward, action_index = self.randomStep()
                        reward += stepReward
                        self.path.append((self.x, self.y))
            else:
                current_distance = abs(self.x - self.currentGoal.x) + abs(self.y - self.currentGoal.y)

                if self.lastGoalDistance is None or current_distance < self.lastGoalDistance:
                    self.lastGoalDistance = current_distance
                    self.goalStepsWithoutProgress = 0
                else:
                    self.goalStepsWithoutProgress += 1

                if self.goalStepsWithoutProgress >= self.maxGoalSteps:
                    self.currentGoal = None
                    self.goalStepsWithoutProgress = 0
                    self.lastGoalDistance = None
                    continue

                action_index = dqnSim.select_action(state, dqnSim.epsilon)

                reward = self.step(action_index)

            self.path.append((self.x, self.y))

            episodeReward += reward

            next_state = dqnSim.getState(self)

            # Store transition in memory
            dqnSim.memory.append((state, action_index, reward, next_state))

            # Optimize model
            dqnSim.optimize_model()

            stepsDone += 1

        return episodeReward, self.path


    def randomStep(self):
        index = random.randint(0, 3)
        return self.step(index), index

    def reset(self):
        self.behavior = set()
        self.path = []
        self.fitness = 0.0
        self.randomStepNum = 0
        self.carrying = False
        self.delivered = 0
        self.goalStepsWithoutProgress = 0
        self.lastGoalDistance = None
        self.delivery = set()
        self.resources = set()
        self.done = False
        self.currentGoal = None
        newx, newy = self.amb.random_valid_position()
        self.setPosition(newx, newy)


    def distance_to_goal(self, x, y):
        return (abs(x - self.currentGoal.x) + abs(y - self.currentGoal.y))/ self.amb.size

    def choose_closest(self, goals):
        if not goals:
            return None
        return min(goals, key=lambda obj: abs(obj.x - self.x) + abs(obj.y - self.y))


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
