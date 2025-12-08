import random
import copy
import numpy as np
from Aprendizagem.EstrategiaAprendizagem import EstrategiaAprendizagem
from Aprendizagem import NeuralNetwork

class EstrategiaGenetica(EstrategiaAprendizagem):
    def __init__(
        self,
        populacao_tamanho: int = 50,
        taxa_mutacao: float = 0.01,
        num_geracoes: int = 25,
        tamanho_torneio: int = 3,
        elitismo_frac: float = 0.2,
        nn_arch=(4, 4, (16, 8)),
        passos_por_avaliacao: int = 500,
        mutation_std: float = 0.1,
    ):
        self.indice_atual = None
        self.populacao_tamanho = int(populacao_tamanho)
        self.taxa_mutacao = float(taxa_mutacao)
        self.num_geracoes = int(num_geracoes)
        self.tamanho_torneio = int(tamanho_torneio)
        self.elitismo_frac = float(elitismo_frac)
        self.nn_arch = nn_arch
        self.passos_por_avaliacao = int(passos_por_avaliacao)
        self.mutation_std = float(mutation_std)

        template_nn = NeuralNetwork.create_network_architecture(*self.nn_arch)
        self.num_weights = template_nn.compute_num_weights()

        self.populacao = [
            np.random.uniform(-1.0, 1.0, self.num_weights).astype(np.float32)
            for _ in range(self.populacao_tamanho)
        ]

        self.fitness = np.zeros(self.populacao_tamanho, dtype=np.float64)

        self.treinada = False
        self.best_individuo = None
        self.best_neural_network = None

    def _selecionar_torneio(self):
        competitors = random.sample(range(self.populacao_tamanho), self.tamanho_torneio)
        best = max(competitors, key=lambda i: self.fitness[i])
        return best

    def _crossover(self, w1: np.ndarray, w2: np.ndarray):
        if self.num_weights <= 1:
            return w1.copy(), w2.copy()
        point = random.randint(1, self.num_weights - 1)
        child1 = np.concatenate([w1[:point], w2[point:]]).astype(np.float32)
        child2 = np.concatenate([w2[:point], w1[point:]]).astype(np.float32)
        return child1, child2

    def _mutar(self, w: np.ndarray) -> np.ndarray:
        w = w.copy()
        mask = np.random.rand(self.num_weights) < self.taxa_mutacao
        if mask.any():
            noise = np.random.normal(0.0, self.mutation_std, size=self.num_weights)
            w[mask] += noise[mask]
        return w

    def _avaliar_individuo(self, pesos: np.ndarray, ambiente) -> float:
        nn = NeuralNetwork.create_network_architecture(*self.nn_arch)
        nn.load_weights(pesos)

        class _EvalAgent:
            def __init__(self, ambiente_ref):
                self.ambiente = ambiente_ref
                if hasattr(self.ambiente, "agentx") and hasattr(self.ambiente, "agenty"):
                    self.x = getattr(self.ambiente, "agentx")
                    self.y = getattr(self.ambiente, "agenty")
                elif hasattr(self.ambiente, "pos_agent_inicio"):
                    self.x, self.y = self.ambiente.pos_agent_inicio
                else:
                    self.x, self.y = 0, 0

                self.recompensa_total = 0.0
                self.foundGoal = False
                self.path = [(self.x, self.y)]

            def percepcao_inputs(self):
                size = getattr(self.ambiente, "size", None)
                if size is None:
                    if hasattr(self.ambiente, "largura"):
                        size = max(1, self.ambiente.largura)
                    else:
                        size = 100.0
                return np.array([
                    self.x / float(size),
                    self.y / float(size),
                    getattr(self.ambiente, "goalx", 0) / float(size),
                    getattr(self.ambiente, "goaly", 0) / float(size)
                ], dtype=np.float32)

            def step_with_nn(self, nn_model):
                inputs = self.percepcao_inputs()
                out = nn_model.forward(inputs)
                action_index = int(np.argmax(out))
                action_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                dx, dy = action_map[action_index]

                newx = self.x + dx
                newy = self.y + dy

                size = getattr(self.ambiente, "size", None)
                if size is None:
                    if hasattr(self.ambiente, "largura"):
                        size_x = self.ambiente.largura
                    else:
                        size_x = 100
                    if hasattr(self.ambiente, "altura"):
                        size_y = self.ambiente.altura
                    else:
                        size_y = 100
                    in_bounds = (0 <= newx < size_x and 0 <= newy < size_y)
                else:
                    in_bounds = (0 <= newx < size and 0 <= newy < size)

                if not in_bounds:
                    self.recompensa_total += -1.0
                    return False

                obj = None
                if hasattr(self.ambiente, "get_object_here"):
                    obj = self.ambiente.get_object_here(newx, newy)

                GoalClass = getattr(self.ambiente, "GoalClass", None)
                WallClass = getattr(self.ambiente, "WallClass", None)

                if obj is not None:
                    if WallClass is not None and isinstance(obj, WallClass):
                        self.recompensa_total += -1.0
                        return False
                    if GoalClass is not None and isinstance(obj, GoalClass):
                        self.recompensa_total += 1000.0
                        self.x, self.y = newx, newy
                        self.path.append((self.x, self.y))
                        self.foundGoal = True
                        return True

                prev_dist = None
                if hasattr(self.ambiente, "goalx") and hasattr(self.ambiente, "goaly"):
                    prev_dist = abs(self.x - self.ambiente.goalx) + abs(self.y - self.ambiente.goaly)
                self.x, self.y = newx, newy
                self.path.append((self.x, self.y))
                new_dist = None
                if hasattr(self.ambiente, "goalx") and hasattr(self.ambiente, "goaly"):
                    new_dist = abs(self.x - self.ambiente.goalx) + abs(self.y - self.ambiente.goaly)

                reward = 1
                if (prev_dist is not None) and (new_dist is not None) and (new_dist < prev_dist):
                    reward += 2
                self.recompensa_total += reward
                return False

        if hasattr(ambiente, "clone"):
            sim_env = ambiente.clone()
        else:
            sim_env = copy.deepcopy(ambiente)

        eval_agent = _EvalAgent(sim_env)

        if hasattr(sim_env, "random_valid_position"):
            start_x, start_y = sim_env.random_valid_position()
            eval_agent.x, eval_agent.y = start_x, start_y
            eval_agent.path = [(start_x, start_y)]

        for _ in range(self.passos_por_avaliacao):
            done = eval_agent.step_with_nn(nn)
            if getattr(eval_agent, "foundGoal", False):
                break
            if hasattr(sim_env, "atualizacao"):
                sim_env.atualizacao()

        return float(eval_agent.recompensa_total)

    def run(self, ambiente, verbose: bool = True):
        pop = self.populacao_tamanho

        for gen in range(self.num_geracoes):
            self.fitness = np.zeros(pop, dtype=np.float64)

            for i in range(pop):
                pesos = self.populacao[i]
                fitness = self._avaliar_individuo(pesos, ambiente)
                self.fitness[i] = fitness

            if verbose:
                avg_f = float(np.mean(self.fitness))
                best_f = float(np.max(self.fitness))
                best_idx = int(np.argmax(self.fitness))
                print(f"[GA] Geração {gen+1}/{self.num_geracoes} | Avg fitness: {avg_f:.3f} | Best: {best_f:.3f} (idx {best_idx})")

            n_elite = max(1, int(self.elitismo_frac * pop))
            idx_sorted = np.argsort(self.fitness)[::-1]  # decrescente
            elites = [self.populacao[int(idx)] for idx in idx_sorted[:n_elite]]
            nova_pop = elites.copy()
            while len(nova_pop) < pop:
                p1 = self._selecionar_torneio()
                p2 = self._selecionar_torneio()
                child1, child2 = self._crossover(self.populacao[p1], self.populacao[p2])
                child1 = self._mutar(child1)
                child2 = self._mutar(child2)
                nova_pop.append(child1)
                if len(nova_pop) < pop:
                    nova_pop.append(child2)

            self.populacao = nova_pop[:pop]

        if np.all(self.fitness == 0):
            for i in range(pop):
                self.fitness[i] = self._avaliar_individuo(self.populacao[i], ambiente)

        best_idx = int(np.argmax(self.fitness))
        self.best_individuo = self.populacao[best_idx].copy()
        best_nn = NeuralNetwork.create_network_architecture(*self.nn_arch)
        best_nn.load_weights(self.best_individuo)
        self.best_neural_network = best_nn
        self.treinada = True

        if verbose:
            print(f"[GA] Treino concluído. Melhor idx: {best_idx} | fitness: {float(self.fitness[best_idx]):.3f}")

    def escolher_acao(self, estado, acoes_possiveis):
        index_to_action = {0: "cima", 1: "baixo", 2: "esquerda", 3: "direita"}

        if self.treinada and self.best_neural_network is not None:
            size = getattr(estado, "size", None)
            if size is None:
                if hasattr(estado, "largura"):
                    size = max(1, estado.largura)
                else:
                    size = 100.0

            inputs = np.array([
                estado.posicao[0] / float(size),
                estado.posicao[1] / float(size),
                getattr(estado, "goalx", 0) / float(size),
                getattr(estado, "goaly", 0) / float(size)
            ], dtype=np.float32)

            out = self.best_neural_network.forward(inputs)
            idx = int(np.argmax(out))
            return index_to_action.get(idx, "cima")

        individuo = self.populacao[self.indice_atual] if 0 <= self.indice_atual < len(self.populacao) else self.populacao[0]
        temp_nn = NeuralNetwork.create_network_architecture(*self.nn_arch)
        temp_nn.load_weights(individuo)
        size = getattr(estado, "size", None)
        if size is None:
            if hasattr(estado, "largura"):
                size = max(1, estado.largura)
            else:
                size = 100.0

        inputs = np.array([
            estado.posicao[0] / float(size),
            estado.posicao[1] / float(size),
            getattr(estado, "goalx", 0) / float(size),
            getattr(estado, "goaly", 0) / float(size)
        ], dtype=np.float32)

        out = temp_nn.forward(inputs)
        idx = int(np.argmax(out))
        return index_to_action.get(idx, "cima")

    def salvar_melhor(self, path="best_weights.npy"):
        if self.best_individuo is not None:
            np.save(path, self.best_individuo)

    def carregar_melhor(self, path="best_weights.npy"):
        w = np.load(path)
        self.best_individuo = w
        nn = NeuralNetwork.create_network_architecture(*self.nn_arch)
        nn.load_weights(w)
        self.best_neural_network = nn
        self.treinada = True
