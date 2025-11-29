import random
from collections import deque

import numpy as np
from numpy import argmax

import AmbienteFarol
import NeuralNetwork


class DQN:
    def __init__(self, inputSize, outputSize):
        self.nn = NeuralNetwork.create_network_architecture(inputSize, outputSize, (16,8))

    def forward(self, input):
        return self.nn.forward(input)

    def compute_gradients(self, x, target):
        return self.nn.compute_gradients(x, target)

amb = AmbienteFarol.Farol("Farol.txt")

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 16
target_update_freq = 100
memory_size = 10000
episodes = 25
action_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Initialize Q-networks
inputSize = 4
outputSize = 4

policy_nn = DQN(inputSize, outputSize)

weights = policy_nn.nn.weights

target_nn = DQN(inputSize, outputSize)
target_nn.nn.load_weights(policy_nn.nn.weights.copy())

optimizer = NeuralNetwork.Adam(policy_nn.nn.weights, lr = learning_rate)
memory = deque(maxlen=memory_size)


def select_action(state, epsilon):
    if random.random() < epsilon:
        action_index = random.randint(0, 3)  # integer index
    else:
        q_values = policy_nn.forward(state)
        action_index = argmax(q_values)

    move = action_map[action_index]  # convert index to move for amb
    return move, action_index


# Function to optimize the model using experience replay
def optimize_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)
    actions = np.array(actions)  # actions should be indices 0..output_size-1

    # Compute current Q-values for the actions taken
    q_values = np.array([policy_nn.forward(s) for s in states])  # shape: [batch_size, output_size]
    q_values_taken = q_values[np.arange(batch_size), actions]  # Q(s, a) for each action taken

    # Compute target Q-values using target network
    next_q_values = np.array([target_nn.forward(s) for s in next_states])
    max_next_q_values = np.max(next_q_values, axis=1)
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute the gradient of the loss w.r.t weights
    grads = np.zeros_like(policy_nn.nn.weights)

    for i in range(batch_size):
        target = q_values[i].copy()
        target[actions[i]] = target_q_values[i]  # Only update the taken action
        grad_i = policy_nn.compute_gradients(states[i], target)
        grads += grad_i / batch_size  # average gradient

    # Update weights using Adam optimizer
    optimizer.step(grads)
    policy_nn.nn.load_weights(optimizer.params)

rewards_per_episode = []
steps_done = 0

for episode in range(episodes):
    amb.reset()
    state = np.array([amb.agentx / amb.size,                # Posição do agente normalizada
                      amb.agenty / amb.size,
                      (amb.goalx - amb.agentx) / amb.size,   # Posição do agente em relação ao goal normalizada
                      (amb.goaly - amb.agenty) / amb.size])
    print(state)
    episode_reward = 0
    done = False
    print(f"Episode {episode + 1} start")
    max_steps = 2500
    steps = 0
    while not done and steps < max_steps:
        # Select action
        move, action_index = select_action(state, epsilon)
        _ , reward, done = amb.step(action_index)
        next_state = np.array([
                    amb.agentx / amb.size,
                    amb.agenty / amb.size,
                    (amb.goalx - amb.agentx) / amb.size,
                    (amb.goaly - amb.agenty) / amb.size])
        # Store transition in memory
        memory.append((state, action_index, reward, next_state, done))

        # Update state
        state = next_state
        episode_reward += reward

        # Optimize model
        optimize_model()

        # Update target network periodically
        if steps_done > 0 and steps_done % target_update_freq == 0:
            target_nn.nn.load_weights(policy_nn.nn.weights)

        steps_done += 1
        steps += 1
        if done:
            print(f"Goal reached in {steps} steps!")
            break
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    rewards_per_episode.append(episode_reward)


    print(f"Episode {episode + 1} reward: {episode_reward}")
    print(f"Distance to goal {amb.distance_to_goal()}\n")

# Plotting the rewards per episode
import matplotlib.pyplot as plt

plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN')
plt.show()