import random
from Projeto.Objetos.ObjetosAmbiente import Goal, Wall, Ground, DeliveryPoint, Resource

class AgenteAprendizagem:

    def __init__(self, amb, neural_network):
        self.randomStepNum = 0
        self.y = None
        self.x = None
        self.yInitial = None
        self.xInitial = None
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.num_steps = 1500
        self.amb = amb
        self.neural_network = neural_network
        self.weights = neural_network.weights.copy()
        self.behavior = set()
        self.path = []
        self.fitness = 0.0
        self.resources = set()
        self.delivery = set()
        self.carrying = False
        self.currentGoal = None
        self.delivered = 0
        self.done = False
        self.obs = None
        self.foundGoal = False
    def observacao(self, depth):
        observation = self.amb.observacaoPara(self, depth=depth)
        obs = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for i in range(4):  # for each direction
            dx, dy = directions[i]  # get the direction vector from your action map

            for step in range(1, depth + 1):  # dynamic range based on depth
                obj = observation[i][step - 1]

                if isinstance(obj, Goal):
                    obs.append(0.9)
                elif isinstance(obj, DeliveryPoint):
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
                    newx = self.x + dx * step
                    newy = self.y + dy * step
                    if self.currentGoal is not None and self.amb.type == "Farol":
                        newDist = self.distanceToGoal(newx, newy)
                        if self.distanceToGoal(self.x, self.y) > newDist:
                            obs.append(0.5)  # moving closer
                        else:
                            obs.append(0.1)
                    else:
                        obs.append(0.1)
                else:
                    obs.append(-0.9)  # None is outside
        self.obs = obs

    def update_goal_from_observation(self):
        if self.currentGoal is None:
            if not self.carrying and self.resources:
                self.currentGoal = self.closest(self.resources)
            elif self.carrying and self.delivery:
                self.currentGoal = self.closest(self.delivery)

    def closest(self, objects):
        if not objects:
            return None
        return min( objects, key=lambda obj: abs(obj.x - self.x) + abs(obj.y - self.y))

    def setPosition(self, x, y):
        self.x = x
        self.y = y
        self.yInitial = y
        self.xInitial = x

    def mutate(self, mutation_rate):
        self.weights = self.neural_network.weights
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += random.uniform(-0.1, 0.1)

        self.neural_network.load_weights(self.weights)

    def setAmbiente(self, ambiente):
        self.amb = ambiente


    def getFitness(self):
        return self.fitness

    def distanceToGoal(self, x, y):
        if self.currentGoal is None and self.amb.type == "Farol":
            self.currentGoal = Goal(self.amb.goalx, self.amb.goaly)
        return (abs(x - self.currentGoal.x) + abs(y - self.currentGoal.y)) / self.amb.size


    def randomStep(self):
        index = random.randint(0, 3)
        reward, done = self.amb.agir(index, self)
        return  reward, index, done

    def age(self, action_index):
        reward, done = self.amb.agir(action_index, self)
        self.avaliacaoEstadoAtual(reward)
        return reward, done

    def avaliacaoEstadoAtual(self, reward):
        self.fitness += reward

    def reset(self):
        self.behavior = set()
        self.path = []
        self.fitness = 0.0
        self.randomStepNum = 0
        self.carrying = False
        self.delivered = 0
        self.delivery = set()
        self.resources = set()
        self.done = False
        self.currentGoal = None
        newx, newy = self.amb.random_valid_position()
        self.setPosition(newx, newy)

