import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches
from numpy import argmax
import Agent
import AmbienteFarol
import NeuralNetwork


class Dqn_nn:
    def __init__(self, inputSize, outputSize):
        self.nn = NeuralNetwork.create_network_architecture(inputSize, outputSize, (16,16))

    def forward(self, input):
        return self.nn.forward(input)

    def compute_gradients(self, x, target):
        return self.nn.compute_gradients(x, target)

class DqnSimulation:

    def __init__(self):

        self.optimization_steps = 0
        self.amb = AmbienteFarol.Farol("Farol.txt")

        # Hyperparameters
        learning_rate = 0.002
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.985
        self.batch_size = 32
        self.target_update_freq = 50
        memory_size = 50000
        self.episodes = 500
        self.action_map = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        # Initialize Q-networks
        inputSize = 15
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
        dones = np.array(dones, dtype=np.float32)
        actions = np.array(actions)

        # Get current Q-values
        current_q = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            q_vals = self.policy_nn.forward(states[i])
            current_q[i] = q_vals[actions[i]]

        # Get target Q-values
        target_q = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i] = rewards[i]
            else:
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

        base_state = [
            agent.x / agent.farol.size,
            agent.y / agent.farol.size,
            agent.distance_to_goal(agent.x, agent.y)
        ]
        surroundings = agent.surroundings()
        return np.array([*base_state, *surroundings])

    def dqnRun(self):
        self.optimization_steps = 0
        rewards_per_episode = []
        pathsPerEpisode = []
        for episode in range(self.episodes):

            print(f"Episode {episode + 1} start")

            agent = Agent.AgenteLearner(self.amb, None)

            self.amb.reset()

            #Agent sim
            episode_reward, path = agent.run_DQN_simulation(None, None, self)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)


            rewards_per_episode.append(episode_reward)
            pathsPerEpisode.append(path)

            # Add periodic logging
            if episode % 5 == 0:
                print(f"Epsilon: {self.epsilon:.3f}, Memory: {len(self.memory)}")

            print(f"Episode {episode + 1} reward: {episode_reward}")
            print(f"Distance to goal {agent.distance_to_goal_agent()}\n")

        # Plotting the rewards per episode
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DQN')

        ##### PATH PLOT #####
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = cm.rainbow(np.linspace(0, 1, len(pathsPerEpisode)))

        # Plot walls and goal
        for wall in self.amb.walls:
            ax.add_patch(
                patches.Rectangle(
                    (wall.x - 0.5, wall.y - 0.5),  # lower-left corner
                    1, 1,  # width, height
                    facecolor='black'
                )
            )

        ax.text(self.amb.goalx, self.amb.goaly, "G", color='green', fontsize=9, ha='center', va='center',
                fontweight='bold')
        # Plot paths
        plot_gens = [0,50,100,150,200,250,300,350,400,450,499]
        for i in plot_gens:
            path = pathsPerEpisode[i]
            avg_fitness = rewards_per_episode[i]  # Get the avg combined fitness
            x_vals = [p[0] for p in path]
            y_vals = [p[1] for p in path]
            ax.plot(x_vals, y_vals, color=colors[i], label=f"Gen {i + 1} (Avg Fitness: {avg_fitness:.2f})",
                    alpha=0.7)
            ax.plot(x_vals[-1], y_vals[-1], 'x', color=colors[i], markersize=10, markeredgewidth=2)

        ax.set_xlim(-1, 100)
        ax.set_ylim(-1, 100)
        ax.set_title("Best Agent Paths Over Generations")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)

        plt.show()
