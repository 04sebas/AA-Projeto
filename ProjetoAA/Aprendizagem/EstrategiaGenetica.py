import random
import numpy as np
from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Aprendizagem.RedeNeuronal import create_network_architecture


class EstrategiaGenetica:
    def __init__(
        self,
        populacao_tamanho=100,
        taxa_mutacao=0.01,
        num_geracoes=25,
        tamanho_torneio=3,
        elitismo_frac=0.2,
        nn_arch=(15, 4, (16, 8)),
        passos_por_avaliacao=1000,
        mutation_std=0.1,
    ):
        self.indice_atual = None
        self.populacao_tamanho = int(populacao_tamanho)
        self.taxa_mutacao = float(taxa_mutacao)
        self.num_geracoes = int(num_geracoes)
        self.tamanho_torneio = int(tamanho_torneio)
        self.elitismo_frac = float(elitismo_frac)
        self.nn_arch = tuple(nn_arch)
        self.passos_por_avaliacao = int(passos_por_avaliacao)
        self.mutation_std = float(mutation_std)

        self.num_weights = None
        self.populacao = None
        self.fitness = None
        self.treinada = False
        self.best_individuo = None
        self.best_neural_network = None

    def _selecionar_torneio(self):
        competitors = random.sample(range(self.populacao_tamanho), self.tamanho_torneio)
        best = max(competitors, key=lambda i: self.fitness[i])
        return best

    def _crossover(self, w1, w2):
        if self.num_weights <= 1:
            return w1.copy(), w2.copy()
        point = random.randint(1, self.num_weights - 1)
        child1 = np.concatenate([w1[:point], w2[point:]]).astype(np.float32)
        child2 = np.concatenate([w2[:point], w1[point:]]).astype(np.float32)
        return child1, child2

    def _mutate_weights(self, weights):
        w = weights.copy()
        mask = np.random.rand(self.num_weights) < self.taxa_mutacao
        if np.any(mask):
            w[mask] += np.random.randn(np.sum(mask)).astype(np.float32) * self.mutation_std
        return w

    def _init_population(self):
        nn_proto = create_network_architecture(*self.nn_arch)
        self.num_weights = int(nn_proto.compute_num_weights())
        pop = []
        for _ in range(self.populacao_tamanho):
            w = np.random.uniform(-1.0, 1.0, size=(self.num_weights,)).astype(np.float32)
            pop.append(w)
        self.populacao = pop
        self.fitness = np.zeros(self.populacao_tamanho, dtype=np.float32)

    def run(self, ambiente, verbose=True):
        self._init_population()
        best_paths_per_gen = []
        avg_fitness_per_gen = []
        last_generation = []
        n_elite = max(1, int(self.elitismo_frac * self.populacao_tamanho))

        if verbose:
            print(f"[GA] População: {self.populacao_tamanho}, Gerações: {self.num_geracoes}, Elite: {n_elite}")

        for gen in range(self.num_geracoes):
            if verbose:
                print(f"[GA] Geração {gen + 1}/{self.num_geracoes}")

            per_generation_agents = []

            for i, weights in enumerate(self.populacao):
                nn = create_network_architecture(*self.nn_arch)
                nn.load_weights(weights)
                agent = AgenteAprendizagem()
                agent.neural_network = nn
                agent.weights = weights.copy()

                start = ambiente.posicao_aleatoria()
                ambiente.posicoes[agent] = tuple(start)
                agent.pos = tuple(start)
                agent.path = [agent.pos]
                agent.found_goal = False

                fitness = 0.0

                for step in range(self.passos_por_avaliacao):
                    obs = ambiente.observacao_para(agent)
                    agent.observacao(obs)

                    acc = agent.age()

                    try:
                        reward = ambiente.agir(acc, agent)
                        if reward is None:
                            reward = 0.0
                    except Exception as e:
                        if verbose:
                            print(f"[GA] Erro ao executar agir (ind={i}): {e}")
                        reward = -1.0

                    fitness += float(reward)
                    agent.path.append(agent.pos)

                    if getattr(agent, "found_goal", False):
                        break

                self.fitness[i] = fitness
                per_generation_agents.append(agent)

            order = np.argsort(-self.fitness)
            self.populacao = [self.populacao[idx] for idx in order]
            self.fitness = self.fitness[order]
            per_generation_agents = [per_generation_agents[idx] for idx in order]

            best_fit = float(self.fitness[0])
            avg_fit = float(np.mean(self.fitness))
            avg_fitness_per_gen.append(avg_fit)

            best_agent = per_generation_agents[0]
            best_paths_per_gen.append(best_agent.path)

            last_generation = per_generation_agents  # agentes ordenados da última avaliação

            if verbose:
                print(f"[GA] Gen {gen + 1}: best={best_fit:.2f}, avg={avg_fit:.2f}")

            new_pop = []
            for k in range(n_elite):
                new_pop.append(self.populacao[k].copy())

            while len(new_pop) < self.populacao_tamanho:
                parent_idx1 = self._selecionar_torneio()
                parent_idx2 = self._selecionar_torneio()
                w1 = self.populacao[parent_idx1].copy()
                w2 = self.populacao[parent_idx2].copy()

                child1, child2 = self._crossover(w1, w2)
                child1 = self._mutate_weights(child1)
                if len(new_pop) < self.populacao_tamanho:
                    new_pop.append(child1)
                if len(new_pop) < self.populacao_tamanho:
                    child2 = self._mutate_weights(child2)
                    new_pop.append(child2)

            self.populacao = new_pop
            self.fitness = np.zeros(self.populacao_tamanho, dtype=np.float32)

        best_weights = self.populacao[0].copy()
        best_nn = create_network_architecture(*self.nn_arch)
        best_nn.load_weights(best_weights)

        self.best_individuo = 0
        self.best_neural_network = best_nn
        self.treinada = True

        if verbose:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import matplotlib.cm as cm
                fig, ax = plt.subplots(figsize=(10, 10))

                walls = getattr(ambiente, "paredes", None) or getattr(ambiente, "walls", None)
                if walls:
                    for wall in walls:
                        if isinstance(wall, tuple) and len(wall) >= 2:
                            wx, wy = wall[0], wall[1]
                        else:
                            wx, wy = getattr(wall, "x", None), getattr(wall, "y", None)
                        if wx is not None and wy is not None:
                            ax.add_patch(patches.Rectangle((wx - 0.5, wy - 0.5), 1, 1, facecolor='black'))

                goalx = getattr(ambiente, "goalx", None)
                goaly = getattr(ambiente, "goaly", None)
                if goalx is None or goaly is None:
                    goal_pos = getattr(ambiente, "goal_pos", None) or getattr(ambiente, "pos_farol", None) or getattr(
                        ambiente, "goal", None)
                    if isinstance(goal_pos, tuple) and len(goal_pos) >= 2:
                        goalx, goaly = goal_pos[0], goal_pos[1]
                    elif hasattr(goal_pos, "x") and hasattr(goal_pos, "y"):
                        goalx, goaly = goal_pos.x, goal_pos.y

                if goalx is not None and goaly is not None:
                    ax.text(goalx, goaly, "G", color='green', fontsize=9, ha='center', va='center', fontweight='bold')

                total_gens = len(best_paths_per_gen)
                if total_gens == 0:
                    total_gens = 1
                plot_gens = list(range(min(24, total_gens)))
                if total_gens - 1 not in plot_gens:
                    plot_gens.append(total_gens - 1)

                colors = cm.rainbow(np.linspace(0, 1, len(plot_gens)))

                for idx_plot, i in enumerate(plot_gens):
                    if i < 0 or i >= len(best_paths_per_gen):
                        continue
                    path = best_paths_per_gen[i]
                    if not path:
                        continue
                    x_vals = [p[0] for p in path]
                    y_vals = [p[1] for p in path]
                    avg_label = avg_fitness_per_gen[i] if i < len(avg_fitness_per_gen) else 0.0
                    ax.plot(x_vals, y_vals, label=f"Geração {i + 1} (Avg Fit: {avg_label:.2f})", alpha=0.7,
                            color=colors[idx_plot])
                    ax.plot(x_vals[-1], y_vals[-1], 'x', markersize=8, color=colors[idx_plot])

                largura = getattr(ambiente, "largura", None)
                altura = getattr(ambiente, "altura", None)
                if largura is not None and altura is not None:
                    ax.set_xlim(-1, largura + 1)
                    ax.set_ylim(-1, altura + 1)
                else:
                    ax.set_xlim(-1, 100)
                    ax.set_ylim(-1, 100)

                ax.set_title("Evolução dos Melhores Caminhos por Geração")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.grid(True)
                ax.legend()
                plt.tight_layout()

                plt.figure(figsize=(10, 5))
                plt.plot(range(len(avg_fitness_per_gen)), avg_fitness_per_gen, marker='o')
                plt.title("Fitness Médio por Geração")
                plt.xlabel("Geração")
                plt.ylabel("Fitness Médio")
                plt.grid(True)
                plt.tight_layout()

                fig2, ax2 = plt.subplots(figsize=(10, 10))
                if walls:
                    for wall in walls:
                        if isinstance(wall, tuple) and len(wall) >= 2:
                            wx, wy = wall[0], wall[1]
                        else:
                            wx, wy = getattr(wall, "x", None), getattr(wall, "y", None)
                        if wx is not None and wy is not None:
                            ax2.add_patch(patches.Rectangle((wx - 0.5, wy - 0.5), 1, 1, facecolor='black'))

                if goalx is not None and goaly is not None:
                    ax2.text(goalx, goaly, "G", color='green', fontsize=10, ha='center', va='center', fontweight='bold')

                cmap = plt.get_cmap("viridis")
                if last_generation:
                    colors2 = cmap(np.linspace(0, 1, len(last_generation)))
                    for i, agent in enumerate(last_generation):
                        if hasattr(agent, "path") and len(agent.path) > 1:
                            xs = [p[0] for p in agent.path]
                            ys = [p[1] for p in agent.path]
                            ax2.plot(xs, ys, color=colors2[i], alpha=0.5)
                            ax2.plot(xs[-1], ys[-1], 'x', color=colors2[i], markersize=6)

                if largura is not None and altura is not None:
                    ax2.set_xlim(-1, largura + 1)
                    ax2.set_ylim(-1, altura + 1)
                else:
                    ax2.set_xlim(-1, 100)
                    ax2.set_ylim(-1, 100)

                ax2.set_title("Todos os Caminhos – Última Geração")
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")
                ax2.grid(True)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"[GA] Erro ao gerar gráficos: {e}")

        return best_weights, best_nn


