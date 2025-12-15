import random
import numpy as np
from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
from ProjetoAA.Ambientes.AmbienteFarol import AmbienteFarol
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
        dummy_agent = AgenteAprendizagem(politica={"alcance": 3})
        if hasattr(ambiente, "get_action_names"):
            dummy_agent.set_action_space(ambiente.get_action_names())
        input_size = dummy_agent.get_input_size()
        num_actions = len(dummy_agent.nomes_accao)
        available_actions = dummy_agent.nomes_accao
        sensor_range = getattr(dummy_agent.sensores, "alcance", 3)
        self.nn_arch = (input_size, num_actions, self.nn_arch[2] if len(self.nn_arch) > 2 else (16, 8))

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
                ambiente.reset()
                nn = create_network_architecture(*self.nn_arch)
                nn.load_weights(weights)

                agent = AgenteAprendizagem(politica={"alcance": sensor_range}, nomes_accao=available_actions)
                agent.neural_network = nn
                agent.weights = weights.copy()

                agent.path = []
                agent.found_goal = False
                agent.recompensa_total = 0.0
                agent.recursos_recolhidos = 0
                agent.recursos_depositados = 0

                if isinstance(ambiente, AmbienteFarol):
                    start = ambiente.posicao_aleatoria_treino()
                else:
                    start = [0,0]

                ambiente.posicoes[agent] = tuple(start)
                agent.pos = tuple(start)
                agent.path = [agent.pos]
                fitness = 0.0

                for step in range(self.passos_por_avaliacao):
                    obs = ambiente.observacao_para(agent)
                    agent.observacao(obs)
                    acc = agent.age()

                    reward = ambiente.agir(acc, agent) or None
                    fitness += float(reward)
                    agent.path.append(agent.pos)

                    done_flag = False
                    if ambiente.nome == "AmbienteFarol":
                        done_flag = bool(getattr(agent, "found_goal", False) or (ambiente.posicoes.get(agent) == getattr(ambiente, "pos_farol", None)))
                    elif ambiente.nome == "AmbienteForaging":
                        done_flag = ambiente.terminou([agent])

                    if done_flag:
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
            last_generation = per_generation_agents

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
                else:
                    recursos_list = []

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

                fig, ax = plt.subplots(figsize=(10, 10))

                for (rx, ry), info in recursos_list:
                    ax.add_patch(
                        patches.Circle((rx, ry), radius=0.4, facecolor="gold", alpha=0.6, edgecolor='k', linewidth=0.3))
                    q = info.get("quantidade", "")
                    ax.text(rx, ry, f"{q}", color="black", ha="center", va="center", fontsize=7)

                for nx, ny in ninhos_list:
                    ax.add_patch(patches.Circle((nx, ny), radius=0.5, facecolor="blue", edgecolor='k'))
                    ax.text(nx, ny, "N", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

                for ox, oy in obstaculos_list:
                    ax.add_patch(patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black"))

                total_gens = len(best_paths_per_gen)
                if total_gens == 0:
                    total_gens = 1
                plot_gens = list(range(min(24, total_gens)))
                if total_gens - 1 not in plot_gens:
                    plot_gens.append(total_gens - 1)

                colors = cm.rainbow(_np.linspace(0, 1, len(plot_gens)))

                for idx_plot, i in enumerate(plot_gens):
                    if i < 0 or i >= len(best_paths_per_gen):
                        continue
                    path = best_paths_per_gen[i]
                    if not path:
                        continue
                    xs = [p[0] for p in path]
                    ys = [p[1] for p in path]
                    avg_label = avg_fitness_per_gen[i] if i < len(avg_fitness_per_gen) else 0.0
                    ax.plot(xs, ys, label=f"Geração {i + 1} (Avg Fit: {avg_label:.2f})", alpha=0.7,
                            color=colors[idx_plot])
                    ax.plot(xs[-1], ys[-1], 'x', markersize=8, color=colors[idx_plot])

                ax.set_xlim(-1, larg + 1)
                ax.set_ylim(-1, alt + 1)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title("Evolução dos Melhores Caminhos por Geração")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.grid(True)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc='upper right', fontsize='small', ncol=1)

                plt.tight_layout()

                plt.figure(figsize=(10, 4.5))
                plt.plot(range(len(avg_fitness_per_gen)), avg_fitness_per_gen, marker='o')
                plt.title("Fitness Médio por Geração")
                plt.xlabel("Geração")
                plt.ylabel("Fitness Médio")
                plt.grid(True)
                plt.tight_layout()

                fig2, ax2 = plt.subplots(figsize=(10, 10))

                for (rx, ry), info in recursos_list:
                    ax2.add_patch(
                        patches.Circle((rx, ry), radius=0.4, facecolor="gold", alpha=0.6, edgecolor='k', linewidth=0.3))
                    q = info.get("quantidade", "")
                    ax2.text(rx, ry, f"{q}", color="black", ha="center", va="center", fontsize=7)

                for nx, ny in ninhos_list:
                    ax2.add_patch(patches.Circle((nx, ny), radius=0.5, facecolor="blue", edgecolor='k'))
                    ax2.text(nx, ny, "N", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

                for ox, oy in obstaculos_list:
                    ax2.add_patch(patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black"))

                cmap = plt.get_cmap("viridis")
                if last_generation:
                    colors2 = cmap(_np.linspace(0, 1, len(last_generation)))
                    for i, agent in enumerate(last_generation):
                        if hasattr(agent, "path") and len(agent.path) > 1:
                            xs = [p[0] for p in agent.path]
                            ys = [p[1] for p in agent.path]
                            ax2.plot(xs, ys, color=colors2[i], alpha=0.5)
                            ax2.plot(xs[-1], ys[-1], 'x', color=colors2[i], markersize=6)

                ax2.set_xlim(-1, larg + 1)
                ax2.set_ylim(-1, alt + 1)
                ax2.set_aspect('equal', adjustable='box')
                ax2.set_title("Todos os Caminhos – Última Geração")
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")
                ax2.grid(True)
                plt.tight_layout()

                plt.show()

            except Exception as e:
                print(f"[GA] Erro ao gerar gráficos: {e}")

        return best_weights, best_nn



