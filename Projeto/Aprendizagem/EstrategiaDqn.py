import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches
from numpy import argmax

from Projeto.Ambientes.AmbienteFarol import AmbienteFarol
from Projeto.Aprendizagem import NeuralNetwork
from Projeto.Ambientes.AmbienteRecolecao import AmbienteRecolecao
from Projeto.Agentes.AgenteAprendizagem import AgenteAprendizagem
from Projeto.Objetos.ObjetosAmbiente import Goal


class Dqn_nn:
    def __init__(self, inputSize, outputSize):
        self.nn = NeuralNetwork.create_network_architecture(inputSize, outputSize, (16, 16))
        self.weigths = self.nn.weights.copy()
    def forward(self, input):
        return self.nn.forward(input)

    def compute_gradients(self, x, target):
        return self.nn.compute_gradients(x, target)

class EstrategiaDqn:

    def __init__(self, input):

        self.optimization_steps = 0

        # Hyperparameters
        learning_rate = 0.002
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.985
        self.batch_size = 32
        self.target_update_freq = 50
        memory_size = 50000
        self.episodes = 100
        self.action_map = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        # Initialize Q-networks
        inputSize = input
        outputSize = 4

        self.policy_nn = Dqn_nn(inputSize, outputSize)

        self.weights = self.policy_nn.nn.weights

        self.target_nn = Dqn_nn(inputSize, outputSize)
        self.target_nn.nn.load_weights(self.policy_nn.nn.weights.copy())

        self.optimizer = NeuralNetwork.Adam(self.policy_nn.nn.weights, lr=learning_rate)
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            action_index = random.randint(0, 3)  # integer index
        else:
            q_values = self.policy_nn.forward(state)
            action_index = argmax(q_values)

        return action_index

    # Function to optimize the model using experience replay
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        actions = np.array(actions)

        # Get current Q-values
        current_q = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            q_vals = self.policy_nn.forward(states[i])
            current_q[i] = q_vals[actions[i]]

        # Get target Q-values
        target_q = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            next_q_vals = self.target_nn.forward(next_states[i])
            target_q[i] = rewards[i] + self.gamma * np.max(next_q_vals)

        # Compute gradients
        grads = np.zeros_like(self.policy_nn.nn.weights)

        for i in range(self.batch_size):
            state = states[i]
            action = actions[i]

            # Forward pass to get current Q-values
            q_values = self.policy_nn.forward(state)

            # Create target vector
            target = q_values.copy()
            target[action] = target_q[i]

            # Compute gradient for this sample
            grad_i = self.policy_nn.compute_gradients(state, target)
            grads += grad_i

        # Average gradients
        grads = grads / self.batch_size

        # Update weights
        self.optimizer.step(grads)
        self.policy_nn.nn.load_weights(self.optimizer.params)

        # Update target network periodically
        self.optimization_steps += 1
        if self.optimization_steps % self.target_update_freq == 0:
            self.target_nn.nn.load_weights(self.policy_nn.nn.weights.copy())

    def getState(self, agent):

        x = agent.x / agent.amb.size
        y = agent.y / agent.amb.size

        agent.observacao(3)

        obs = agent.obs

        goalx, goaly = 0, 0

        if agent.currentGoal is not None:
            goalx = agent.currentGoal.x / agent.amb.size
            goaly = agent.currentGoal.y / agent.amb.size


        if agent.amb.type == "Recolecao":
            return np.array([x, y, *obs, goalx, goaly])
        else:
            return np.array([x, y, *obs, agent.distanceToGoal(agent.x, agent.y)])

    def dqnRun(self, fileName, type):
        self.optimization_steps = 0
        rewards_per_episode = []
        pathsPerEpisode = []
        num_steps = 1500
        amb = None
        for episode in range(self.episodes):

            if type == "Farol":
                amb = AmbienteFarol(fileName)
            elif type == "Recolecao":
                amb = AmbienteRecolecao(fileName)
            else:
                print("Type incorrect.")
                return
            print(f"Episode {episode + 1} start")
            amb.reset()

            #Agent sim
            ##ALGORITMO
            stepsDone = 0
            episodeReward = 0
            path = []
            agent = AgenteAprendizagem(amb, self.policy_nn)
            x, y = amb.random_valid_position()
            agent.setPosition(x, y)
            done = False

            if amb.type == "Farol":
                agent.currentGoal = Goal(agent.amb.goalx, agent.amb.goaly)

            for _ in range(num_steps):


                agent.observacao(3)  # Observa antes de random step, caso haja algum goal
                agent.update_goal_from_observation()
                state = self.getState(agent)
                reward = 0
                action_index = 0

                if agent.currentGoal is None:
                    if not agent.carrying:
                        if agent.resources:
                            goals = list(agent.resources)
                            agent.currentGoal = agent.closest(goals)

                        else:
                            agent.randomStepNum += 1
                            stepReward, action_index, done = agent.randomStep()
                            reward += stepReward
                            agent.path.append((agent.x, agent.y))
                    else:
                        if agent.delivery:
                            goals = list(agent.delivery)
                            agent.currentGoal = agent.closest(goals)

                        else:
                            agent.randomStepNum += 1
                            stepReward, action_index, done = agent.randomStep()
                            reward += stepReward
                            agent.path.append((agent.x, agent.y))
                else:

                    current_distance = abs(agent.x - agent.currentGoal.x) + abs(agent.y - agent.currentGoal.y)
                    if agent.amb.type == "Recolecao":
                        if agent.amb.lastGoalDistance is None or current_distance < agent.amb.lastGoalDistance:
                            agent.amb.lastGoalDistance = current_distance
                            agent.amb.goalStepsWithoutProgress = 0
                        else:
                            agent.amb.goalStepsWithoutProgress += 1

                        if agent.amb.goalStepsWithoutProgress >= agent.amb.maxGoalSteps:
                            agent.amb.currentGoal = None
                            agent.amb.goalStepsWithoutProgress = 0
                            agent.amb.lastGoalDistance = None
                            continue
                    action_index = self.select_action(state, self.epsilon)

                    reward, done = agent.age(action_index)

                path.append((agent.x, agent.y))

                episodeReward += reward

                next_state = self.getState(agent)

                # Store transition in memory
                self.memory.append((state, action_index, reward, next_state, done))

                # Optimize model
                self.optimize_model()

                stepsDone += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            rewards_per_episode.append(episodeReward)
            pathsPerEpisode.append(path)

            # Add periodic logging
            if episode % 5 == 0:
                print(f"Epsilon: {self.epsilon:.3f}, Memory: {len(self.memory)}")

            print(f"Episode {episode + 1} reward: {episodeReward}")

            if agent.amb.type == "Recolecao":
                print(f"Random steps per episode: {agent.randomStepNum}")
                print(f"Resources delivered: {agent.delivered}\n")
            else:
                print(f"Distance to goal: {agent.distanceToGoal(agent.x, agent.y)}")
            if done:
                print(f"Episode {episode + 1} done, found goal in: {stepsDone}")
        # Plotting the rewards per episode
        window = 10
        avg_rewards = [
            np.mean(rewards_per_episode[i:i + window])
            for i in range(0, len(rewards_per_episode), window)
        ]

        plt.plot(avg_rewards)
        plt.xlabel('Episode (x5)')
        plt.ylabel('Average Reward')
        plt.title('DQN (Average Reward per 50 Episodes)')
        plt.show()

        ##### PATH PLOT #####
        # Plot walls and goal
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = cm.rainbow(np.linspace(0, 1, len(pathsPerEpisode)))

        # Plot walls and goal
        for wall in amb.walls:
            ax.add_patch(
                patches.Rectangle(
                    (wall.x - 0.5, wall.y - 0.5),  # lower-left corner
                    1, 1,  # width, height
                    facecolor='black'
                )
            )

        for resource in amb.resources:
            ax.text(resource.x, resource.y, "R", color='green', fontsize=9, ha='center', va='center',
                    fontweight='bold')

        for delivery in amb.deliveryPoints:
            ax.text(delivery.x, delivery.y, "P", color='red', fontsize=9, ha='center', va='center',
                    fontweight='bold')

        for agente in amb.agentes:
            ax.text(agente.x, agente.y, "A", color='blue', fontsize=9, ha='center', va='center', fontweight='bold')

        # Plot paths
        plot_gens = [0,100,200, 300, 400,499]
        for i in plot_gens:
            path = pathsPerEpisode[i]
            avg_fitness = rewards_per_episode[i]  # Get the avg combined fitness
            x_vals = [p[0] for p in path]
            y_vals = [p[1] for p in path]
            ax.plot(x_vals, y_vals, color=colors[i], label=f"Gen {i + 1} (Avg Fitness: {avg_fitness:.2f})",
                    alpha=0.7)
            ax.plot(x_vals[-1], y_vals[-1], 'x', color=colors[i], markersize=10, markeredgewidth=2)

        ax.set_xlim(-1, 50)
        ax.set_ylim(-1, 50)
        ax.set_title("Best Agent Paths Over Episodes")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)
        plt.show()
