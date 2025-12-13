import json
import numpy as np
import sns
from matplotlib import pyplot as plt
from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
from ProjetoAA.Agentes.AgenteFixo import AgenteFixo
from ProjetoAA.Ambientes.AmbienteFarol import AmbienteFarol
from ProjetoAA.Ambientes.AmbienteForaging import AmbienteForaging
from ProjetoAA.Aprendizagem.EstrategiaGenetica import EstrategiaGenetica
from ProjetoAA.Aprendizagem.EstrategiaQLearning import EstrategiaDQN


def _salvar_melhor_nn(ambiente, agent_index, weights, nn_obj, nn_arch=None, out_dir="models"):
    import os, pickle

    os.makedirs(out_dir, exist_ok=True)

    env_name = getattr(ambiente, "nome", type(ambiente).__name__)
    filename = f"{env_name}_agente{agent_index}_nn.pkl"
    path = os.path.join(out_dir, filename)

    meta = {
        "env_name": env_name,
        "agent_index": agent_index,
        "nn_arch": nn_arch if nn_arch is not None else getattr(nn_obj, "arch", None),
    }

    data = {
        "meta": meta,
        "weights_flat": weights,
        "hidden_weights": getattr(nn_obj, "hidden_weights", None),
        "hidden_biases": getattr(nn_obj, "hidden_biases", None),
        "output_weights": getattr(nn_obj, "output_weights", None),
        "output_bias": getattr(nn_obj, "output_bias", None),
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"[SALVO] NN guardada em {path}")
    return path, meta

class MotorDeSimulacao:
    def __init__(self):
        self._ficheiro_config = None
        self.recursos_iniciais = None
        self.ambiente = None
        self.passos = 0
        self.agentes = []
        self.ordem = []
        self.ativo = False
        self.max_passos = 1000
        self.visualizacao = True
        self.historico_agentes = {}
        self.historico_recompensa = {}
        self.config_estrategias = []

    def lista_agentes(self):
        return self.agentes

    def cria(self, nome_do_ficheiro_parametros):
        try:
            with open(nome_do_ficheiro_parametros, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._cria_ambiente(config.get("ambiente", {}))
            self.config_estrategias = config.get("estrategia", [])
            self._cria_agentes(config.get("agentes", []))
            config_simulador = config.get("simulador", {})
            self.max_passos = config_simulador.get("max_passos", 100)
            self.visualizacao = config_simulador.get("visualizacao", True)
            self.ativo = True
            self._ficheiro_config = nome_do_ficheiro_parametros
            print(f"Simulação criada com sucesso: {len(self.agentes)} agentes no ambiente")
        except FileNotFoundError:
            print(f"Erro: Ficheiro {nome_do_ficheiro_parametros} não encontrado.")
        except json.JSONDecodeError:
            print(f"Erro: Ficheiro JSON inválido: {nome_do_ficheiro_parametros}")
        except Exception as e:
            print(f"Erro a criar simulação: {e}")
        return self

    def _cria_ambiente(self, config_ambiente):
        tipo_ambiente = config_ambiente.get("tipo", "AmbienteFarol")

        if tipo_ambiente == "AmbienteFarol":
            self.ambiente = AmbienteFarol(
                largura=config_ambiente.get("largura", 100),
                altura=config_ambiente.get("altura", 100),
                pos_farol=tuple(config_ambiente.get("pos_farol", [50, 75])),
                obstaculos=config_ambiente.get("obstaculos", [])
            )
        elif tipo_ambiente == "AmbienteForaging":
            self.ambiente = AmbienteForaging(
                largura=config_ambiente.get("largura", 100),
                altura=config_ambiente.get("altura", 100),
                recursos=config_ambiente.get("recursos", []),
                ninhos=config_ambiente.get("ninhos", []),
                obstaculos=config_ambiente.get("obstaculos", [])
            )
        else:
            raise ValueError(f"Tipo de ambiente desconhecido: {tipo_ambiente}")

    def _obter_config_estrategia(self, tipo_estrategia):
        for e in self.config_estrategias:
            if e.get("tipo") == tipo_estrategia:
                return e
        return {}

    def _cria_agentes(self, config_agentes):
        for config in config_agentes:
            tipo = config.get("tipo")
            quantidade = config.get("quantidade", 1)
            posicoes = config.get("posicao_inicial", "random")
            trainable = config.get("trainable", True)

            for i in range(quantidade):
                if posicoes == "random":
                    pos = self.ambiente.posicao_aleatoria()
                else:
                    if isinstance(posicoes, list) and i < len(posicoes):
                        pos = posicoes[i]
                    else:
                        pos = posicoes
                if isinstance(pos, list):
                    pos = tuple(pos)
                if tipo == "AgenteFixo":
                    agente = AgenteFixo(posicao=list(pos), politica=config.get("politica", {}))
                elif tipo == "AgenteAprendizagem":
                    tipo_estrategia = config.get("tipo_estrategia", config.get("estrategia", {}).get("nome", "qlearning"))
                    agente = AgenteAprendizagem(nome=f"Aprendiz_{i}", politica=config.get("politica", {}), posicao=pos)
                    agente.tipo_estrategia = tipo_estrategia
                    agente.estrategia_conf = config.get("estrategia", config.get("estrategia_conf", {}))
                else:
                    raise ValueError(f"Tipo de agente desconhecido: {tipo}")

                agente.trainable = bool(trainable)
                agente.pos = list(pos)
                self.agentes.append(agente)
                print(f"Criado agente {agente.nome} na posição {pos}")

    def carregar_rede(self, filepath, agente_idx):
        import pickle, os
        from ProjetoAA.Aprendizagem.RedeNeuronal import RedeNeuronal, relu, output_fn

        base = os.path.dirname(__file__)
        fullpath = os.path.join(base, filepath)

        if not os.path.exists(fullpath):
            raise FileNotFoundError(f"Não encontrado: {fullpath}")

        with open(fullpath, "rb") as f:
            data = pickle.load(f)

        meta = data["meta"]
        nn_arch = meta["nn_arch"]

        input_size, output_size, hidden_arch = nn_arch

        nn = RedeNeuronal(input_size, output_size, hidden_arch, relu, output_fn)

        nn.hidden_weights = data.get("hidden_weights")
        nn.hidden_biases = data.get("hidden_biases")
        nn.output_weights = data.get("output_weights")
        nn.output_bias = data.get("output_bias")

        self.agentes[agente_idx].neural_network = nn
        self.agentes[agente_idx].weights = data.get("weights_flat")

        print(f"[LOAD] NN carregada para o agente {agente_idx}")

    def fase_treino(self):
        saved_files = {}

        for idx, agente in enumerate(self.agentes):
            if not getattr(agente, "trainable", False):
                continue

            tipo = getattr(agente, "tipo_estrategia", None)
            conf = getattr(agente, "estrategia_conf", None)
            if conf is None:
                conf = self._obter_config_estrategia(tipo or "genetica")

            available_actions = ["cima", "baixo", "direita", "esquerda"]
            if getattr(self.ambiente, "recursos", None):
                if "recolher" not in available_actions:
                    available_actions.append("recolher")
            if getattr(self.ambiente, "ninhos", None):
                if "depositar" not in available_actions:
                    available_actions.append("depositar")

            if hasattr(agente, "set_action_space"):
                agente.set_action_space(available_actions)
            else:
                agente.nomes_accao = available_actions

            try:
                input_size = agente.get_input_size()
            except Exception:
                alcance = getattr(agente.sensores, "alcance", 3)
                input_size = (2 * alcance + 1) ** 2 - 1 + 3

            output_size = len(available_actions)

            if tipo == "genetica":
                ga = EstrategiaGenetica(
                    populacao_tamanho=conf.get("populacao_tamanho", 100),
                    taxa_mutacao=conf.get("taxa_mutacao", 0.01),
                    num_geracoes=conf.get("num_ger", 25),
                    tamanho_torneio=conf.get("tamanho_torneio", 2),
                    elitismo_frac=conf.get("elitismo_frac", 0.1),
                    nn_arch=(input_size, output_size, conf.get("hidden", (16, 8))),
                    passos_por_avaliacao=conf.get("passos_por_avaliacao", 750),
                    mutation_std=conf.get("mutation_std", 0.1),
                )

                self.ambiente.posicoes = {}

                best_weights, best_nn = ga.run(self.ambiente, verbose=True)

                self.ambiente.reset()

                agente.neural_network = best_nn
                agente.weights = np.array(best_weights).copy()

                path, meta = _salvar_melhor_nn(
                    self.ambiente,
                    idx,
                    best_weights,
                    best_nn,
                    nn_arch=getattr(ga, "nn_arch", None)
                )
                meta = dict(meta or {})
                meta["tipo"] = "genetica"
                saved_files[idx] = {"path": path, "meta": meta}
                print(f"[fase_treino] Agente {idx}: rede genetica aplicada e salva em {path}")

            elif tipo == "dqn":
                dqn_conf = conf or {}
                dqn = EstrategiaDQN(
                    nn_arch=(input_size, output_size, dqn_conf.get("hidden", (16, 8))),
                    episodes=dqn_conf.get("episodes", 100),
                    passos_por_ep=dqn_conf.get("passos_por_ep", 750),
                    gamma=dqn_conf.get("gamma", 0.99),
                    epsilon=dqn_conf.get("epsilon", 0.90),
                    epsilon_min=dqn_conf.get("epsilon_min", 0.1),
                    epsilon_decay=dqn_conf.get("epsilon_decay", 0.95),
                    batch_size=dqn_conf.get("batch_size", 32),
                    target_update_freq=dqn_conf.get("target_update_freq", 50),
                    memory_size=dqn_conf.get("memory_size", 50000),
                    learning_rate=dqn_conf.get("learning_rate", 0.001),
                )

                self.ambiente.posicoes = {}

                best_weights, best_nn = dqn.run(self.ambiente, verbose=True)

                self.ambiente.reset()

                agente.neural_network = best_nn
                agente.weights = np.array(best_weights).copy()

                path, meta = _salvar_melhor_nn(
                    self.ambiente,
                    idx,
                    best_weights,
                    best_nn,
                    nn_arch=getattr(dqn, "nn_arch", None)
                )
                meta = dict(meta or {})
                meta["tipo"] = "dqn"
                saved_files[idx] = {"path": path, "meta": meta}
                print(f"[fase_treino] Agente {idx}: rede dqn aplicada e salva em {path}")

            else:
                continue

        return saved_files

    def fase_teste(self):
        self.executa(self.max_passos)

    def executa(self, max_passos=None):
        if not self.ativo:
            print("Simulação não foi criada corretamente.")
            return

        max_p = max_passos if max_passos is not None else self.max_passos
        self.ambiente.posicoes = {}

        for agente in self.agentes:
            if not hasattr(agente, "pos") or agente.pos is None:
                pos = self.ambiente.posicao_aleatoria()
                agente.pos = list(pos)
            self.ambiente.posicoes[agente] = tuple(agente.pos)

        self.historico_agentes = {}
        self.historico_recompensa = {}

        for agente in self.agentes:
            self.historico_agentes[agente] = [list(agente.pos)]
            nome = getattr(agente, "nome", f"Agente_{self.agentes.index(agente)}")
            self.historico_recompensa[nome] = [0]
        self.recursos_iniciais = {}

        if isinstance(self.ambiente.recursos, dict):
            for pos, info in self.ambiente.recursos.items():
                self.recursos_iniciais[pos] = {"valor": info["valor"], "quantidade": info["quantidade"]}

        for passo in range(max_p):
            todos_terminaram = False
            for agente in self.agentes:
                observacao = self.ambiente.observacao_para(agente)
                agente.observacao(observacao)
                accao = agente.age()
                recompensa = self.ambiente.agir(accao, agente)
                agente.avaliacao_estado_atual(recompensa)

                if isinstance(agente.pos, tuple):
                    agente.pos = list(agente.pos)
                self.historico_agentes[agente].append(list(agente.pos))
                nome = getattr(agente, "nome", f"Agente_{self.agentes.index(agente)}")
                self.historico_recompensa[nome].append(recompensa)

            if hasattr(self.ambiente, "atualizacao"):
                self.ambiente.atualizacao()
            self.passos += 1

            if isinstance(self.ambiente, AmbienteFarol):
                todos_terminaram = all(getattr(ag, "found_goal", False) for ag in self.agentes)

            if todos_terminaram:
                print(f"Simulação terminada no passo {passo}!")
                break

        if self.visualizacao:
            try:
                self.visualizar_caminhos()
            except Exception as e:
                print(f"Erro ao desenhar visualizações: {e}")

    def visualizar_caminhos(self):
        altura = self.ambiente.altura
        largura = self.ambiente.largura
        plt.figure(figsize=(10, 10))
        cores = plt.cm.tab20(np.linspace(0, 1, max(1, len(self.historico_agentes))))
        for i, (agente, caminho) in enumerate(self.historico_agentes.items()):
            if not caminho:
                continue
            xs = [p[0] for p in caminho]
            ys = [p[1] for p in caminho]
            plt.plot(xs, ys, marker='.', linewidth=1, color=cores[i], label=f"{getattr(agente, 'nome', f'Agente {i}')}")
            # start / end markers
            plt.scatter(xs[0], ys[0], c=[cores[i]], marker='o', s=80)  # início
            plt.scatter(xs[-1], ys[-1], c=[cores[i]], marker='x', s=80)  # fim

        # Plotar recursos (suporta dict ou lista)
        recursos = getattr(self.ambiente, "recursos", None)
        recursos_items = []
        if recursos:
            if isinstance(recursos, dict):
                recursos_items = list(recursos.items())
            else:
                # lista de objetos/dicts com "pos", "quantidade", "valor"
                recursos_items = []
                for r in recursos:
                    pos = tuple(r.get("pos")) if isinstance(r.get("pos"), (list, tuple)) else None
                    if pos is not None:
                        recursos_items.append((pos, r))

        if recursos_items:
            valores = [info.get("valor", info.get("quantidade", 1)) for (_p, info) in recursos_items]
            quantidades = [info.get("quantidade", 1) for (_p, info) in recursos_items]
            min_val, max_val = min(valores), max(valores)
            span = max(1e-6, max_val - min_val)
            cmap = plt.get_cmap("RdYlBu")
            for (pos, info), val, q in zip(recursos_items, valores, quantidades):
                x, y = pos
                size = max(40, 40 + (q - min(quantidades)) * 4)
                norm = (val - min_val) / span
                color = cmap(norm)
                plt.scatter([x], [y], s=size, marker='o', color=color, alpha=0.8, edgecolors='k')
                plt.text(x + 0.2, y + 0.2, f"q:{q}\nv:{val}", fontsize=7)

        ninhos = getattr(self.ambiente, "ninhos", None)
        if ninhos:
            nx = [p[0] for p in ninhos]
            ny = [p[1] for p in ninhos]
            plt.scatter(nx, ny, s=200, marker='^', color='gold', edgecolors='k', label="Ninhos")

        obs_raw = getattr(self.ambiente, "obstaculos_raw", None)
        if obs_raw:
            obs_x = [p["pos"][0] for p in obs_raw]
            obs_y = [p["pos"][1] for p in obs_raw]
        else:
            obs_set = getattr(self.ambiente, "obstaculos", None)
            if obs_set:
                obs_x = [p[0] for p in obs_set]
                obs_y = [p[1] for p in obs_set]
            else:
                obs_x, obs_y = [], []
        if obs_x:
            plt.scatter(obs_x, obs_y, color="black", marker='s', s=120, label="Obstáculos")

        if hasattr(self.ambiente, "pos_farol"):
            farol_x, farol_y = self.ambiente.pos_farol
            plt.scatter([farol_x], [farol_y], color="red", marker="*", s=300, label="Farol")

        ax = plt.gca()
        ax.set_xlim(-0.5, largura - 0.5)
        ax.set_ylim(-0.5, altura - 0.5)
        ax.set_aspect('equal', adjustable='box')

        xticks = np.arange(0, largura, max(1, largura // 10))
        yticks = np.arange(0, altura, max(1, altura // 10))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajetórias dos Agentes no Ambiente")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    simulador = MotorDeSimulacao().cria("simulador_farol_vazio.json")
    if simulador.ativo:
        """""
        for idx, agente in enumerate(simulador.agentes):
            if not isinstance(agente, AgenteAprendizagem):
                continue
            if not getattr(agente, "trainable", False):
                continue

            try:
                simulador.carregar_rede(f"models/AmbienteFarol_agente0_nn.pkl",idx)
            except FileNotFoundError:
                print(f"[LOAD] NN não encontrada para agente {idx}, será usado aleatório.")
                """""
        simulador.fase_treino()
        simulador.fase_teste()
