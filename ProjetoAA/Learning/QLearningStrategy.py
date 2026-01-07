import math
import random
from collections import deque
import numpy as np
from ProjetoAA.Agents.LearningAgent import LearningAgent
from ProjetoAA.Learning.NeuralNetwork import create_network_architecture, Adam
from ProjetoAA.Objects.Action import Action
from ProjetoAA.Learning.LearningStrategy import LearningStrategy


class QLearningStrategy(LearningStrategy):
    def __init__(
        self,
        nn_arch=(15, 4, (16, 8)),
        episodes=100,
        steps_per_ep=500,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.985,
        batch_size=32,
        target_update_freq=50,
        memory_size=50000,
        learning_rate=0.001,
    ):
        super().__init__(nn_arch=nn_arch, verbose=True)
        self.episodes = int(episodes)
        self.steps_per_ep = int(steps_per_ep)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = float(learning_rate)
        self.optimization_steps = 0
        self.policy = None
        self.target = None
        self.optimizer = None
        
        self.epsilon_history = []

    def _build_networks(self, input_size, output_size, hidden):
        self.policy = DQNNetwork(input_size, output_size, hidden)
        self.target = DQNNetwork(input_size, output_size, hidden)
        self.target.load_weights(self.policy.get_weights().copy())
        self.optimizer = Adam(self.policy.nn.weights, lr=self.learning_rate)

    def choose_action(self, state, possible_actions):
       pass 

    def select_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        expected = getattr(self.policy.nn, "input_size", None)
        if expected is None:
            # Fallback for some reason
            pass
        if state.shape[0] != expected:
            raise ValueError(f"[DQN] state size {state.shape[0]} != network input {expected}")
        if random.random() < self.epsilon:
            return random.randint(0, self.policy.nn.output_size - 1)
        q = self.policy.propagate(state)
        return int(np.argmax(q))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=np.float32)
        actions = np.array(actions, dtype=int)
        
        q_target = np.zeros(self.batch_size, dtype=np.float32)

        # Double DQN Implementation
        for i in range(self.batch_size):
            if dones[i]:
                q_target[i] = rewards[i]
            else:
                # 1. Select best action using Policy Network
                next_q_policy = self.policy.propagate(next_states[i])
                best_action = np.argmax(next_q_policy)
                
                # 2. Evaluate that action using Target Network
                next_q_target = self.target.propagate(next_states[i])
                q_target[i] = rewards[i] + self.gamma * float(next_q_target[best_action])

        gradients = [np.zeros_like(w) for w in self.policy.nn.weights]

        for i in range(self.batch_size):
            state = states[i]
            action = actions[i]
            if state.shape[0] != self.policy.nn.input_size:
                raise RuntimeError(f"nn_input mismatch: {state.shape[0]} vs {self.policy.nn.input_size}")
            q_values = self.policy.propagate(state)
            target = q_values.copy()
            target[action] = q_target[i]
            grad_i = self.policy.compute_gradients(state, target)
            gradients = [g + gi for g, gi in zip(gradients, grad_i)]


        gradients = [g / self.batch_size for g in gradients]
        flat_grads = self.policy.nn.flatten_grads(gradients)
        self.optimizer.step(flat_grads)
        self.policy.nn.unflatten_params(self.optimizer.params)
        self.optimization_steps += 1

        if self.optimization_steps % self.target_update_freq == 0:
            self.target.load_weights(self.policy.get_weights().copy())

    def train(self, environment, verbose=True, range_val=None, training_positions=None):
        self.verbose = verbose
        if range_val is not None:
            sample_agent = LearningAgent(policy={"range": int(range_val)})
        else:
            default_range = getattr(environment, "default_sensor_range", None)
            if default_range is not None:
                sample_agent = LearningAgent(policy={"range": int(default_range)})
            else:
                sample_agent = LearningAgent()
        input_size = sample_agent.get_input_size()
        output_size = len(sample_agent.action_names)
        hidden = self.nn_arch[2] if len(self.nn_arch) >= 3 else (16, 8)
        self._build_networks(input_size, output_size, hidden)

        if self.policy.nn.input_size != input_size:
            raise RuntimeError(
                f"[DQN] inconsistency: policy input {self.policy.nn.input_size} != expected {input_size}")

        self.fitness_history = [] # rewards history
        self.path_history = []
        self.epsilon_history = [] 

        start = [0,0]
        for ep in range(self.episodes):
            if hasattr(environment, "restart"):
                environment.restart()
            elif hasattr(environment, "reset"):
                environment.reset()
            
            if training_positions:
                start = random.choice(training_positions)
            elif hasattr(environment, "random_position"):
                start = environment.random_position()
            else:
                start = [0, 0]

            agent = LearningAgent(policy={"range": sample_agent.sensors.sensing_range})
            agent.pos = start
            # Fix: random start if needed, but keeping logic consistent
            environment.positions[agent] = tuple(agent.pos)
            agent.found_target = False
            obs = environment.observation_for(agent)
            agent.observation(obs)
            state = agent.build_nn_input(obs)
            episode_reward = 0.0
            path = [tuple(agent.pos)]

            for step in range(self.steps_per_ep):
                action_idx = self.select_action(state)
                action_name = agent.action_names[action_idx]
                acc = Action(action_name)

                try:
                    reward = environment.act(acc, agent)
                    if reward is None:
                        reward = 0.0
                except Exception:
                    reward = -1.0

                next_obs = environment.observation_for(agent)
                agent.observation(next_obs)
                next_state = agent.build_nn_input(next_obs)
                state = state.astype(np.float32)
                next_state = next_state.astype(np.float32)
                done_flag = environment.finished([agent])
                
                self.memory.append((state, action_idx, float(reward), next_state, done_flag))
                # Warmup
                warmup = max(self.batch_size * 2, 1000)
                if len(self.memory) >= warmup:
                    self.optimize_model()

                state = next_state
                episode_reward += float(reward)
                path.append(tuple(agent.pos))

                if done_flag:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            self.fitness_history.append(episode_reward)
            self.path_history.append(path)
            self.epsilon_history.append(self.epsilon)

            if self.verbose:
                print(f"[DQN] Ep {ep + 1}/{self.episodes} reward={episode_reward:.2f} eps={self.epsilon:.3f}")
        
        self.best_weights = self.policy.get_weights().copy()
        self.best_nn = self.policy.nn

        # Use generate_plots with callback for extra plots
        self.generate_plots(environment, fitness_title="Reward per Episode", paths_title="Trajectories — DQN", other_plots=self._extra_plots)

        return self.best_weights, self.best_nn

    def _extra_plots(self, plt):
        # Plot Moving Average
        rewards = self.fitness_history
        if len(rewards) > 5:
            plt.figure(figsize=(10, 4.5))
            plt.plot(rewards, alpha=0.3, label="Raw")
            window = max(1, min(20, len(rewards) // 10 or 1))
            if window > 1:
                moving_avg = np.convolve(np.array(rewards), np.ones(window) / window, mode='valid')
                plt.plot(range(window - 1, window - 1 + len(moving_avg)), moving_avg, linewidth=2, label=f'Moving Average (w={window})')
            plt.title("Moving Average of Rewards")
            plt.grid()
            plt.legend()
            plt.tight_layout()

        # Plot Epsilon
        if self.epsilon_history:
            plt.figure(figsize=(8, 3))
            plt.plot(self.epsilon_history, marker='.', label='Epsilon')
            plt.title("Epsilon over training")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.grid(True)
            plt.tight_layout()


class DQNNetwork:
    def __init__(self, input_size, output_size, hidden):
        self.nn = create_network_architecture(input_size, output_size, hidden)
    def propagate(self, x):
        return self.nn.propagate(x)
    def compute_gradients(self, x, target):
        return self.nn.compute_gradients(x, target)
    def load_weights(self, w):
        self.nn.load_weights(w)
    def get_weights(self):
        return self.nn.weights.copy()
