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


def _salvar_melhor_nn(ambiente, agent_index, weights, nn_obj, nn_arch=None, out_dir="models", tipo="genetica"):
    import os, pickle, re, time

    os.makedirs(out_dir, exist_ok=True)

    env_name = getattr(ambiente, "nome", type(ambiente).__name__)
    base_prefix = f"{env_name}_agente{agent_index}_{tipo}"

    pattern = re.compile(rf"^{re.escape(base_prefix)}(?:_v(\d+))?\.pkl$")
    max_v = 0
    for fn in os.listdir(out_dir):
        m = pattern.match(fn)
        if m:
            if m.group(1):
                try:
                    v = int(m.group(1))
                    if v > max_v:
                        max_v = v
                except Exception:
                    pass
            else:
                if max_v < 1:
                    max_v = 1

    next_v = max_v + 1 if max_v > 0 else 1
    filename = f"{base_prefix}_v{next_v}.pkl"

    path = os.path.join(out_dir, filename)

    meta = {
        "env_name": env_name,
        "agent_index": agent_index,
        "nn_arch": nn_arch if nn_arch is not None else getattr(nn_obj, "arch", None),
        "tipo": tipo,
        "timestamp": int(time.time()),
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

    print(f"[SALVO] NN (tipo={tipo}) guardada em {path}")
    return path, meta

def _to_offsets(offsets):
    arr = np.asarray(offsets, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1 and arr.shape[0] == 2:
        return arr.reshape(1, 2)
    if arr.ndim == 1 and arr.shape[0] % 2 == 0:
        return arr.reshape(-1, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    try:
        return arr.reshape(-1, 2)
    except Exception:
        return np.empty((0, 2), dtype=float)


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
                    politica_agente = config.get("politica", {}) or {}
                    tipo_politica = politica_agente.get("tipo", "fixo")
                    nome_agente = f"AgenteFixo_{tipo_politica}_{len(self.agentes)}"
                    agente = AgenteFixo(
                        posicao=list(pos),
                        politica=politica_agente,
                        nome=nome_agente
                    )
                elif tipo == "AgenteAprendizagem":
                    politica_agente = config.get("politica", {}) or {}
                    tipo_estrategia = config.get("tipo_estrategia", "genetica")
                    nome_agente = f"AgenteAprendizagem_{tipo_estrategia}_{len(self.agentes)}"
                    agente = AgenteAprendizagem(
                        nome=nome_agente,
                        politica=politica_agente,
                        posicao=list(pos)
                    )
                    agente.trainable = bool(config.get("trainable", False))
                    agente.tipo_estrategia = tipo_estrategia
                    agente.estrategia_conf = config.get("estrategia_conf", {})
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

        meta = data.get("meta", {})
        nn_arch = meta.get("nn_arch")
        tipo = meta.get("tipo", "unknown")

        nn_obj = None
        try:
            if nn_arch:
                input_size, output_size, hidden_arch = nn_arch
                nn_obj = RedeNeuronal(input_size, output_size, hidden_arch, relu, output_fn)
                if data.get("hidden_weights") is not None:
                    nn_obj.hidden_weights = data.get("hidden_weights")
                if data.get("hidden_biases") is not None:
                    nn_obj.hidden_biases = data.get("hidden_biases")
                if data.get("output_weights") is not None:
                    nn_obj.output_weights = data.get("output_weights")
                if data.get("output_bias") is not None:
                    nn_obj.output_bias = data.get("output_bias")

                if hasattr(nn_obj, "weights") and data.get("weights_flat") is not None:
                    try:
                        nn_obj.weights = data.get("weights_flat")
                    except Exception:
                        pass

        except Exception as e:
            print(f"[LOAD] Aviso: não foi possível reconstruir a RedeNeuronal a partir do meta.nn_arch: {e}")
            nn_obj = None

        agente = self.agentes[agente_idx]

        if nn_obj is not None:
            agente.neural_network = nn_obj
            agente.weights = data.get("weights_flat")
            print(f"[LOAD] NN reconstruída e atribuída ao agente {agente_idx} (tipo={tipo})")
            return

        agente.weights = data.get("weights_flat")
        try:
            nn_existing = getattr(agente, "neural_network", None)
            if nn_existing is not None:
                if data.get("hidden_weights") is not None:
                    setattr(nn_existing, "hidden_weights", data.get("hidden_weights"))
                if data.get("hidden_biases") is not None:
                    setattr(nn_existing, "hidden_biases", data.get("hidden_biases"))
                if data.get("output_weights") is not None:
                    setattr(nn_existing, "output_weights", data.get("output_weights"))
                if data.get("output_bias") is not None:
                    setattr(nn_existing, "output_bias", data.get("output_bias"))
                print(f"[LOAD] Pesos atribuídos à neural_network existente do agente {agente_idx}")
                return
        except Exception:
            pass

        print(f"[LOAD] Apenas weights_flat carregados para agente {agente_idx} (tipo={tipo}).")

    def load_networks(self, pattern: str = None, file_map: dict = None, agents: list = None, verbose: bool = True):
        summary = {}

        if agents is None:
            agents = list(range(len(self.agentes)))

        file_map = file_map or {}

        for idx in agents:
            if idx < 0 or idx >= len(self.agentes):
                summary[idx] = {"status": "error", "msg": "Índice de agente inválido"}
                if verbose:
                    print(f"[LOAD_NETWORKS] Índice inválido: {idx}")
                continue

            agente = self.agentes[idx]

            if not getattr(agente, "trainable", False):
                summary[idx] = {"status": "skipped", "msg": "agente não treinável"}
                if verbose:
                    print(f"[LOAD_NETWORKS] Agente {idx} (não treinável) - skip")
                continue

            filepath = None
            if idx in file_map:
                filepath = file_map[idx]
            elif pattern is not None:
                try:
                    filepath = pattern.format(idx=idx)
                except Exception as e:
                    summary[idx] = {"status": "error", "msg": f"Erro a formatar pattern: {e}"}
                    if verbose:
                        print(f"[LOAD_NETWORKS] Erro a formatar pattern para agente {idx}: {e}")
                    continue
            else:
                summary[idx] = {"status": "skipped", "msg": "nenhum pattern nem file_map fornecido"}
                if verbose:
                    print(f"[LOAD_NETWORKS] Agente {idx} - nenhum ficheiro especificado")
                continue

            try:
                self.carregar_rede(filepath, idx)
                summary[idx] = {"status": "loaded", "msg": filepath}
                if verbose:
                    print(f"[LOAD_NETWORKS] Sucesso: agente {idx} <- {filepath}")
            except FileNotFoundError:
                summary[idx] = {"status": "missing", "msg": filepath}
                if verbose:
                    print(f"[LOAD_NETWORKS] Ficheiro não encontrado para agente {idx}: {filepath}")
            except Exception as e:
                summary[idx] = {"status": "error", "msg": str(e)}
                if verbose:
                    print(f"[LOAD_NETWORKS] Erro ao carregar agente {idx} de {filepath}: {e}")

        return summary

    def autoload_if_flag(self, pattern: str = None, file_map: dict = None, agents: list = None, verbose: bool = True):
        if getattr(self, "autoload_models", False):
            return self.load_networks(pattern=pattern, file_map=file_map, agents=agents, verbose=verbose)
        if verbose:
            print("[autoload_if_flag] autoload_models não activado; nenhum ficheiro carregado.")
        return {}

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
                input_size = (2 * alcance + 1) ** 2 - 1 + 5

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

                best_weights, best_nn = ga.run(self.ambiente, verbose=True,input_size=input_size)

                self.ambiente.reset()

                agente.neural_network = best_nn
                agente.weights = np.array(best_weights).copy()

                path, meta = _salvar_melhor_nn(
                    self.ambiente,
                    idx,
                    best_weights,
                    best_nn,
                    nn_arch=getattr(ga, "nn_arch", None),
                    tipo="genetica"
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

                best_weights, best_nn = dqn.run(self.ambiente, verbose=True,alcance=agente.sensores.alcance)

                self.ambiente.reset()

                agente.neural_network = best_nn
                agente.weights = np.array(best_weights).copy()

                path, meta = _salvar_melhor_nn(
                    self.ambiente,
                    idx,
                    best_weights,
                    best_nn,
                    nn_arch=getattr(dqn, "nn_arch", None),
                    tipo="dqn"
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
            elif isinstance(self.ambiente, AmbienteForaging):
                todos_terminaram = self.ambiente.terminou(self.agentes)

            if todos_terminaram:
                print(f"Simulação terminada no passo {passo}!")
                break

        if self.visualizacao:
            try:
                self.plot_estatisticas_teste()
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

    def plot_estatisticas_teste(self):
        import matplotlib.pyplot as plt
        import numpy as np

        agentes = self.agentes
        nomes = [getattr(ag, "nome", f"Agente_{i}") for i, ag in enumerate(agentes)]

        recolhidos = [getattr(ag, "recursos_recolhidos", 0) for ag in agentes]
        depositados = [getattr(ag, "recursos_depositados", 0) for ag in agentes]
        fitnesses = [getattr(ag, "recompensa_total", 0.0) for ag in agentes]

        x = np.arange(len(nomes))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, recolhidos, width, label="Recolhidos")
        plt.bar(x + width / 2, depositados, width, label="Depositados")
        plt.xticks(x, nomes, rotation=45, ha="right")
        plt.ylabel("Quantidade")
        plt.title("Recursos Recolhidos e Depositados por Agente (Teste)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(x, fitnesses, width * 1.2)
        plt.xticks(x, nomes, rotation=45, ha="right")
        plt.ylabel("Fitness (Recompensa Total)")
        plt.title("Fitness dos Agentes na Fase de Teste")
        plt.tight_layout()
        plt.show()

        print("\nResumo da Fase de Teste:")
        print(f"{'Agente':20s} {'Recolhidos':>10s} {'Depositados':>12s} {'Fitness':>12s}")
        for nome, r, d, f in zip(nomes, recolhidos, depositados, fitnesses):
            print(f"{nome:20s} {r:10d} {d:12d} {f:12.2f}")

    def salvar_animacao_gif(self, filepath, fps=10, trail_len=20):
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter

        if not self.historico_agentes:
            raise RuntimeError("historico_agentes vazio")

        agentes = list(self.historico_agentes.keys())
        paths = [self.historico_agentes[ag] for ag in agentes]
        max_steps = max(len(p) for p in paths)

        fig, ax = plt.subplots(figsize=(8, 8))
        largura, altura = self.ambiente.largura, self.ambiente.altura
        ax.set_xlim(-0.5, largura - 0.5)
        ax.set_ylim(-0.5, altura - 0.5)
        ax.set_aspect("equal")
        ax.grid(True)

        recursos = getattr(self.ambiente, "recursos", {})
        if isinstance(recursos, dict):
            for (x, y), info in recursos.items():
                ax.scatter(x, y, s=80, color="gold", edgecolors="k")

        ninhos = getattr(self.ambiente, "ninhos", [])
        if ninhos:
            nx, ny = zip(*ninhos)
            ax.scatter(nx, ny, s=160, marker="^", color="blue", edgecolors="k")

        colors = plt.cm.tab20(np.linspace(0, 1, len(agentes)))
        agents_scatter = []
        trails = []

        for c in colors:
            sc = ax.scatter([], [], s=80, color=c, edgecolors="k")
            ln, = ax.plot([], [], linewidth=1.5, color=c, alpha=0.7)
            agents_scatter.append(sc)
            trails.append(ln)

        def update(frame):
            for i, path in enumerate(paths):
                if frame < len(path):
                    agents_scatter[i].set_offsets([path[frame]])
                else:
                    agents_scatter[i].set_offsets([])

                start = max(0, frame - trail_len)
                seg = path[start:frame + 1]
                if len(seg) > 1:
                    xs, ys = zip(*seg)
                    trails[i].set_data(xs, ys)
                else:
                    trails[i].set_data([], [])

            ax.set_title(f"Step {frame}")
            return agents_scatter + trails

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=max_steps,
            interval=1000 // fps,
            blit=False
        )

        ani.save(filepath, writer=PillowWriter(fps=fps))
        plt.close(fig)

        print(f"[GIF] Animação salva em {filepath}")

    def run_experiments(self, num_runs: int = 30, max_passos: int = None, file_map: dict = None,agents_to_load: list = None, seed: int = None, save_plot: str = None):
        import numpy as _np
        import random

        if file_map:
            try:
                self.load_networks(file_map=file_map, agents=agents_to_load or list(file_map.keys()))
            except Exception as e:
                print(f"[run_experiments] Aviso ao carregar redes: {e}")

        original_visual = getattr(self, "visualizacao", False)
        self.visualizacao = False

        agent_names = [getattr(a, "nome", f"Agente_{i}") for i, a in enumerate(self.agentes)]
        fitness_runs = []

        is_foraging = "AmbienteForaging" in self.ambiente.nome

        if is_foraging:
            resources_by_agent = {name: {"recolhidos": 0, "depositados": 0} for name in agent_names}
        else:
            successes_by_agent = {name: 0 for name in agent_names}

        for i in range(num_runs):
            if seed is not None:
                s = int(seed) + i
                _np.random.seed(s)
                random.seed(s)

            if hasattr(self.ambiente, "reset"):
                self.ambiente.reset()
            self.ambiente.posicoes = {}

            for agente in self.agentes:
                agente.pos = None
                agente.recompensa_total = 0.0
                agente.found_goal = False

                if hasattr(agente, "recursos_recolhidos"):
                    agente.recursos_recolhidos = 0
                if hasattr(agente, "recursos_depositados"):
                    agente.recursos_depositados = 0

            self.executa(max_passos=max_passos)

            total_fitness = 0.0
            for idx, agente in enumerate(self.agentes):
                nome = getattr(agente, "nome", f"Agente_{idx}")
                total_fitness += float(getattr(agente, "recompensa_total", 0.0) or 0.0)
                if is_foraging:
                    recolh = int(getattr(agente, "recursos_recolhidos", 0) or 0)
                    depos = int(getattr(agente, "recursos_depositados", 0) or 0)
                    resources_by_agent[nome]["recolhidos"] += recolh
                    resources_by_agent[nome]["depositados"] += depos
                else:
                    if bool(getattr(agente, "found_goal", False)):
                        successes_by_agent[nome] += 1

            fitness_runs.append(total_fitness)

            try:
                self.ambiente.posicoes = {}
            except Exception:
                pass

        self.visualizacao = original_visual

        fitness_arr = _np.array(fitness_runs, dtype=float)
        stats = {
            "fitness_mean": float(fitness_arr.mean()) if len(fitness_arr) else 0.0,
            "fitness_std": float(fitness_arr.std()) if len(fitness_arr) else 0.0
        }

        if is_foraging:
            stats["resources_totals"] = {name: {"recolhidos": v["recolhidos"], "depositados": v["depositados"]} for name, v in resources_by_agent.items()}
        else:
            stats["successes_by_agent"] = successes_by_agent

        try:
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(fitness_runs) + 1), fitness_runs, marker="o")
            plt.xlabel("Simulação")
            plt.ylabel("Fitness total")
            plt.title(f"Fitness por simulação (mean={stats['fitness_mean']:.2f}, std={stats['fitness_std']:.2f})")
            plt.grid(True)
            plt.tight_layout()
            if save_plot:
                plt.savefig(save_plot.replace(".png", "_fitness.png"), dpi=200)
            plt.show()
        except Exception as e:
            print(f"[run_experiments] Erro ao desenhar fitness plot: {e}")

        try:
            plt.figure(figsize=(10, 4))
            names = agent_names
            if is_foraging:
                recolhidos = [resources_by_agent[n]["recolhidos"] for n in names]
                depositados = [resources_by_agent[n]["depositados"] for n in names]
                import numpy as _np_local
                x = _np_local.arange(len(names))
                width = 0.35
                plt.bar(x - width / 2, recolhidos, width, label="Recolhidos")
                plt.bar(x + width / 2, depositados, width, label="Depositados")
                plt.xlabel("Agente")
                plt.ylabel("Recursos (total em todas as simulações)")
                plt.title("Recursos recolhidos e depositados por agente")
                plt.xticks(x, names, rotation=45, ha="right")
                plt.legend()
                plt.grid(axis="y")
                plt.tight_layout()
                if save_plot:
                    plt.savefig(save_plot.replace(".png", "_resources_by_agent.png"), dpi=200)
                plt.show()
            else:
                values = [successes_by_agent[n] for n in names]
                plt.bar(names, values)
                plt.xlabel("Agente")
                plt.ylabel("Nº de vezes que chegou ao goal")
                plt.title("Sucessos por agente")
                plt.grid(axis="y")
                plt.tight_layout()
                if save_plot:
                    plt.savefig(save_plot.replace(".png", "_successes_by_agent.png"), dpi=200)
                plt.show()
        except Exception as e:
            print(f"[run_experiments] Erro ao desenhar successes/resources plot: {e}")

        result = {
            "fitness_runs": fitness_runs,
            "summary": stats
        }
        if is_foraging:
            result["resources_by_agent"] = resources_by_agent
        else:
            result["successes_by_agent"] = successes_by_agent

        return result


if __name__ == "__main__":
    simulador = MotorDeSimulacao().cria("simulador_foraging.json")
    if simulador.ativo:
        file_map = {
            2: "models/AmbienteForaging_agente2_genetica_v1.pkl",
            3: "models/AmbienteForaging_agente3_dqn_v1.pkl"
        }
        #resumo = simulador.load_networks(file_map=file_map, agents=[2,3])
        #print(resumo)
        resultados = simulador.run_experiments(num_runs=30, max_passos=750, file_map=file_map, seed=20,save_plot="results/aggregate.png")
        print("Resumo:", resultados["summary"])
        #simulador.fase_treino()
        #simulador.fase_teste()
        #simulador.salvar_animacao_gif("models/trajetorias_foraging.gif", fps=12, trail_len=30)

