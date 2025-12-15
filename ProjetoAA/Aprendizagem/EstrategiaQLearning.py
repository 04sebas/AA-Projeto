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
        epsilon_decay=0.985,
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

        for ep in range(self.episodes):
            if hasattr(ambiente, "reset"):
                ambiente.reset()

            agent = AgenteAprendizagem()
            agent.pos = ambiente.posicao_aleatoria()
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

        if verbose:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import matplotlib.cm as cm
                import numpy as _np

                larg = getattr(ambiente, "largura", 50)
                alt = getattr(ambiente, "altura", 50)

                recursos_raw = getattr(ambiente, "recursos", {})
                ninhos_raw = getattr(ambiente, "ninhos", [])
                obstaculos_raw = getattr(ambiente, "obstaculos", [])

                recursos_list = []
                if isinstance(recursos_raw, dict):
                    for pos, info in recursos_raw.items():
                        recursos_list.append((tuple(pos), dict(info)))
                elif isinstance(recursos_raw, (list, tuple)):
                    for r in recursos_raw:
                        if isinstance(r, dict) and "pos" in r:
                            recursos_list.append(
                                (tuple(r["pos"]), {"quantidade": r.get("quantidade", 1), "valor": r.get("valor", 0)}))

                ninhos_list = []
                if isinstance(ninhos_raw, (list, tuple, set)):
                    for n in ninhos_raw:
                        try:
                            ninhos_list.append(tuple(n))
                        except:
                            pass

                obstaculos_list = []
                if isinstance(obstaculos_raw, (list, tuple, set)):
                    for o in obstaculos_raw:
                        if isinstance(o, tuple) and len(o) >= 2:
                            obstaculos_list.append(tuple(o))
                        elif isinstance(o, list) and len(o) >= 2:
                            obstaculos_list.append(tuple(o))
                        elif isinstance(o, dict) and "pos" in o:
                            obstaculos_list.append(tuple(o["pos"]))

                plt.figure(figsize=(10, 4.5))
                eps_idx = list(range(len(rewards_history)))
                plt.plot(eps_idx, rewards_history, marker='o', label='Reward por episódio', alpha=0.6)

                window = max(1, min(20, len(rewards_history) // 10 or 1))
                if window > 1:
                    movavg = _np.convolve(_np.array(rewards_history), _np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, window - 1 + len(movavg)), movavg, linewidth=2,
                             label=f'Média móvel (w={window})')

                plt.title("Reward por Episódio")
                plt.xlabel("Episódio")
                plt.ylabel("Reward")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                try:
                    eps_history = getattr(self, "epsilon_history", None)
                    if eps_history is None:
                        decay = self.epsilon_decay
                        e = self.epsilon
                        eps_history = []
                        val = 1.0
                        for _ in range(len(rewards_history)):
                            eps_history.append(val)
                            val = max(self.epsilon_min, val * decay)
                    plt.figure(figsize=(8, 3))
                    plt.plot(range(len(eps_history)), eps_history, marker='.', label='Epsilon')
                    plt.title("Epsilon ao longo do treino")
                    plt.xlabel("Episódio")
                    plt.ylabel("Epsilon")
                    plt.grid(True)
                    plt.tight_layout()
                except Exception:
                    pass

                fig, ax = plt.subplots(figsize=(10, 10))

                for (rx, ry), info in recursos_list:
                    ax.add_patch(patches.Circle((rx, ry), radius=0.45, facecolor="gold", alpha=0.8, edgecolor='k',
                                                linewidth=0.3))
                    q = info.get("quantidade", "")
                    ax.text(rx, ry, f"{q}", color="black", ha="center", va="center", fontsize=7)

                for nx, ny in ninhos_list:
                    ax.add_patch(patches.Circle((nx, ny), radius=0.5, facecolor="blue", edgecolor='k'))
                    ax.text(nx, ny, "N", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

                for ox, oy in obstaculos_list:
                    ax.add_patch(patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black"))

                if len(paths) > 0:
                    rewards_arr = _np.array(rewards_history)
                    best_idx = int(_np.argmax(rewards_arr))
                    topk = 6
                    top_indices = list(_np.argsort(-rewards_arr)[:topk])
                    colors = cm.rainbow(_np.linspace(0, 1, len(top_indices)))

                    for ci, idx in enumerate(top_indices):
                        path = paths[idx]
                        if not path:
                            continue
                        xs = [p[0] for p in path]
                        ys = [p[1] for p in path]
                        ax.plot(xs, ys, label=f"Ep {idx} (R={rewards_history[idx]:.1f})", alpha=0.8, linewidth=1.5,
                                color=colors[ci])
                        ax.plot(xs[-1], ys[-1], 'x', markersize=8, color=colors[ci])

                    best_path = paths[best_idx]
                    if best_path:
                        xs = [p[0] for p in best_path]
                        ys = [p[1] for p in best_path]
                        ax.plot(xs, ys, color='red', linewidth=2.5, alpha=0.9,
                                label=f"Melhor Ep {best_idx} (R={rewards_history[best_idx]:.1f})")
                        ax.plot(xs[-1], ys[-1], 'X', color='red', markersize=10)

                try:
                    visit_counts = _np.zeros((alt, larg), dtype=int)
                    for path in paths:
                        for (vx, vy) in path:
                            if 0 <= int(vx) < larg and 0 <= int(vy) < alt:
                                visit_counts[int(vy), int(vx)] += 1
                    vmax = visit_counts.max() if visit_counts.max() > 0 else 1
                    ax.imshow(visit_counts, origin='lower', cmap='hot', alpha=0.35,
                              extent=(-0.5, larg - 0.5, -0.5, alt - 0.5), vmin=0, vmax=vmax)
                except Exception:
                    pass

                ax.set_xlim(-1, larg + 1)
                ax.set_ylim(-1, alt + 1)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title("Trajetórias — DQN (top trajectories & heatmap)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.grid(True)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc='upper right', fontsize='small')

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"[DQN] Erro ao gerar gráficos: {e}")

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