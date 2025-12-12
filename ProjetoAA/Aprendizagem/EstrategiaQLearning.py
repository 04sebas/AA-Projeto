import math
import random
from collections import deque
import numpy as np
from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
from ProjetoAA.Aprendizagem.RedeNeuronal import create_network_architecture, Adam
from ProjetoAA.Objetos.Accao import Accao


class EstrategiaDQN:
    def __init__(
        self,
        nn_arch=(15, 4, (16, 8)),
        episodes=100,
        passos_por_ep=500,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_freq=50,
        memory_size=50000,
        learning_rate=0.001,
    ):
        self.nn_arch = tuple(nn_arch)
        self.episodes = int(episodes)
        self.passos_por_ep = int(passos_por_ep)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = float(learning_rate)
        self.optim_steps = 0
        self.policy = None
        self.target = None
        self.optimizer = None

    def _build_networks(self, input_size, output_size, hidden):
        self.policy = DQNNet(input_size, output_size, hidden)
        self.target = DQNNet(input_size, output_size, hidden)
        self.target.load_weights(self.policy.get_weights().copy())
        self.optimizer = Adam(self.policy.nn.weights, lr=self.learning_rate)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.policy.nn.output_size - 1)
        q = self.policy.forward(state)
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
        current_q = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            qv = self.policy.forward(states[i])
            current_q[i] = qv[actions[i]]

        target_q = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            if dones[i]:
                target_q[i] = rewards[i]
            else:
                next_q = self.target.forward(next_states[i])
                target_q[i] = rewards[i] + self.gamma * float(np.max(next_q))

        grads = [np.zeros_like(w) for w in self.policy.nn.weights]

        for i in range(self.batch_size):
            state = states[i]
            action = actions[i]
            q_values = self.policy.forward(state)
            target = q_values.copy()
            target[action] = target_q[i]
            grad_i = self.policy.compute_gradients(state, target)
            grads = [g + gi for g, gi in zip(grads, grad_i)]


        grads = [g / self.batch_size for g in grads]
        flat_grads = self.policy.nn.flatten_grads(grads)
        self.optimizer.step(flat_grads)
        self.policy.nn.unflatten_params(self.optimizer.params)
        self.optim_steps += 1

        if self.optim_steps % self.target_update_freq == 0:
            self.target.load_weights(self.policy.get_weights().copy())

    def run(self, ambiente, verbose=True):
        sample_agent = AgenteAprendizagem()
        input_size = sample_agent.get_input_size()
        output_size = len(sample_agent.nomes_accao)
        hidden = self.nn_arch[2] if len(self.nn_arch) >= 3 else (16, 8)
        self._build_networks(input_size, output_size, hidden)
        rewards_history = []
        paths = []
        start = ambiente.posicao_aleatoria()
        for ep in range(self.episodes):
            if hasattr(ambiente, "reset"):
                ambiente.reset()

            agent = AgenteAprendizagem()
            agent.pos = list(start)
            ambiente.posicoes[agent] = tuple(agent.pos)
            agent.found_goal = False
            obs = ambiente.observacao_para(agent)
            agent.observacao(obs)
            state = agent.build_nn_input(obs)
            ep_reward = 0.0
            path = [tuple(agent.pos)]

            for step in range(self.passos_por_ep):
                action_idx = self.select_action(state)
                action_name = agent.nomes_accao[action_idx]
                acc = Accao(action_name)

                try:
                    reward = ambiente.agir(acc, agent)
                    if reward is None:
                        reward = 0.0
                except Exception:
                    reward = -1.0

                obs_next = ambiente.observacao_para(agent)
                agent.observacao(obs_next)
                next_state = agent.build_nn_input(obs_next)
                done = getattr(agent, "found_goal", False)
                self.memory.append((state, action_idx, float(reward), next_state, done))
                self.optimize_model()
                state = next_state
                ep_reward += float(reward)
                path.append(tuple(agent.pos))

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(ep_reward)
            paths.append(path)

            if verbose:
                print(f"[DQN] Ep {ep + 1}/{self.episodes} reward={ep_reward:.2f} eps={self.epsilon:.3f}")
        best_weights = self.policy.get_weights().copy()
        return best_weights, self.policy.nn


class DQNNet:
    def __init__(self, input_size, output_size, hidden):
        self.nn = create_network_architecture(input_size, output_size, hidden)
    def forward(self, x):
        return self.nn.forward(x)
    def compute_gradients(self, x, target):
        return self.nn.compute_gradients(x, target)
    def load_weights(self, w):
        self.nn.load_weights(w)
    def get_weights(self):
        return self.nn.weights.copy()