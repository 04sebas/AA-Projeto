import json
import numpy as np
import sns
from matplotlib import pyplot as plt
from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
from ProjetoAA.Agentes.AgenteFixo import AgenteFixo
from ProjetoAA.Ambientes.AmbienteFarol import AmbienteFarol
from ProjetoAA.Ambientes.AmbienteForaging import AmbienteForaging
from ProjetoAA.Aprendizagem.EstrategiaGenetica import EstrategiaGenetica
from ProjetoAA.Aprendizagem.EstrategiaQLearning import EstrategiaQLearning


def _salvar_melhor_rn(ambiente, index_agente, pesos, objeto_rn, arq_rn=None, pasta_saida="models", tipo="genetica"):
    import os, pickle, re, time

    os.makedirs(pasta_saida, exist_ok=True)

    nome_ambiente = getattr(ambiente, "nome", type(ambiente).__name__)
    prefixo_base = f"{nome_ambiente}_agente{index_agente}_{tipo}"

    padrao = re.compile(rf"^{re.escape(prefixo_base)}(?:_v(\d+))?\.pkl$")
    max_v = 0
    for fn in os.listdir(pasta_saida):
        m = padrao.match(fn)
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

    prox_v = max_v + 1 if max_v > 0 else 1
    nome_ficheiro = f"{prefixo_base}_v{prox_v}.pkl"

    caminho = os.path.join(pasta_saida, nome_ficheiro)

    meta = {
        "env_name": nome_ambiente,
        "agent_index": index_agente,
        "nn_arch": arq_rn if arq_rn is not None else getattr(objeto_rn, "arq_rn", None),
        "tipo": tipo,
        "timestamp": int(time.time()),
    }

    dados = {
        "meta": meta,
        "weights_flat": pesos,
        "hidden_weights": getattr(objeto_rn, "pesos_ocultos", None),
        "hidden_biases": getattr(objeto_rn, "vies_ocultos", None),
        "output_weights": getattr(objeto_rn, "pesos_saida", None),
        "output_bias": getattr(objeto_rn, "vies_saida", None),
    }

    with open(caminho, "wb") as f:
        pickle.dump(dados, f)

    print(f"[GUARDADO] RN (tipo={tipo}) guardada em {caminho}")
    return caminho, meta

def _para_offsets(offsets):
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
            treinavel = config.get("trainable", True)

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
                    agente.treinavel = bool(config.get("trainable", False))
                    agente.tipo_estrategia = tipo_estrategia
                    agente.estrategia_conf = config.get("estrategia_conf", {})
                else:
                    raise ValueError(f"Tipo de agente desconhecido: {tipo}")

                agente.treinavel = bool(treinavel)
                agente.pos = list(pos)
                self.agentes.append(agente)
                print(f"Criado agente {agente.nome} na posição {pos}")

    def carregar_rede(self, caminho_ficheiro, indice_agente):
        import pickle, os
        from ProjetoAA.Aprendizagem.RedeNeuronal import RedeNeuronal, relu, funcao_saida

        base = os.path.dirname(__file__)
        caminho_completo = os.path.join(base, caminho_ficheiro)

        if not os.path.exists(caminho_completo):
            raise FileNotFoundError(f"Não encontrado: {caminho_completo}")

        with open(caminho_completo, "rb") as f:
            dados = pickle.load(f)

        meta = dados.get("meta", {})
        arq_rn = meta.get("nn_arch")
        tipo = meta.get("tipo", "desconhecido")

        obj_rn = None
        try:
            if arq_rn:
                tamanho_entrada, tamanho_saida, arq_oculta = arq_rn
                obj_rn = RedeNeuronal(tamanho_entrada, tamanho_saida, arq_oculta, relu, funcao_saida)
                if dados.get("hidden_weights") is not None:
                    obj_rn.pesos_ocultos = dados.get("hidden_weights")
                if dados.get("hidden_biases") is not None:
                    obj_rn.vies_ocultos = dados.get("hidden_biases")
                if dados.get("output_weights") is not None:
                    obj_rn.pesos_saida = dados.get("output_weights")
                if dados.get("output_bias") is not None:
                    obj_rn.vies_saida = dados.get("output_bias")

                if hasattr(obj_rn, "pesos") and dados.get("weights_flat") is not None:
                    try:
                        obj_rn.pesos = dados.get("weights_flat")
                    except Exception:
                        pass

        except Exception as e:
            print(f"[CARREGAR] Aviso: não foi possível reconstruir a RedeNeuronal a partir do meta.nn_arch: {e}")
            obj_rn = None

        agente = self.agentes[indice_agente]

        if obj_rn is not None:
            agente.rede_neuronal = obj_rn
            agente.pesos = dados.get("weights_flat")
            print(f"[CARREGAR] RN reconstruída e atribuída ao agente {indice_agente} (tipo={tipo})")
            return

        agente.pesos = dados.get("weights_flat")
        try:
            rn_existente = getattr(agente, "rede_neuronal", None)
            if rn_existente is not None:
                if dados.get("hidden_weights") is not None:
                    setattr(rn_existente, "pesos_ocultos", dados.get("hidden_weights"))
                if dados.get("hidden_biases") is not None:
                    setattr(rn_existente, "vies_ocultos", dados.get("hidden_biases"))
                if dados.get("output_weights") is not None:
                    setattr(rn_existente, "pesos_saida", dados.get("output_weights"))
                if dados.get("output_bias") is not None:
                    setattr(rn_existente, "vies_saida", dados.get("output_bias"))
                print(f"[CARREGAR] Pesos atribuídos à rede_neuronal existente do agente {indice_agente}")
                return
        except Exception:
            pass

        print(f"[CARREGAR] Apenas weights_flat carregados para agente {indice_agente} (tipo={tipo}).")

    def carregar_redes(self, padrao: str = None, mapa_ficheiros: dict = None, agentes: list = None, detalhado: bool = True):
        resumo = {}

        if agentes is None:
            agentes = list(range(len(self.agentes)))

        mapa_ficheiros = mapa_ficheiros or {}

        for idx in agentes:
            if idx < 0 or idx >= len(self.agentes):
                resumo[idx] = {"status": "erro", "msg": "Índice de agente inválido"}
                if detalhado:
                    print(f"[CARREGAR_REDES] Índice inválido: {idx}")
                continue

            agente = self.agentes[idx]

            if not getattr(agente, "treinavel", False):
                resumo[idx] = {"status": "ignorado", "msg": "agente não treinável"}
                if detalhado:
                    print(f"[CARREGAR_REDES] Agente {idx} (não treinável) - ignorado")
                continue

            caminho_ficheiro = None
            if idx in mapa_ficheiros:
                caminho_ficheiro = mapa_ficheiros[idx]
            elif padrao is not None:
                try:
                    caminho_ficheiro = padrao.format(idx=idx)
                except Exception as e:
                    resumo[idx] = {"status": "erro", "msg": f"Erro a formatar padrão: {e}"}
                    if detalhado:
                        print(f"[CARREGAR_REDES] Erro a formatar padrão para agente {idx}: {e}")
                    continue
            else:
                resumo[idx] = {"status": "ignorado", "msg": "nenhum padrão nem mapa_ficheiros fornecido"}
                if detalhado:
                    print(f"[CARREGAR_REDES] Agente {idx} - nenhum ficheiro especificado")
                continue

            try:
                self.carregar_rede(caminho_ficheiro, idx)
                resumo[idx] = {"status": "carregado", "msg": caminho_ficheiro}
                if detalhado:
                    print(f"[CARREGAR_REDES] Sucesso: agente {idx} <- {caminho_ficheiro}")
            except FileNotFoundError:
                resumo[idx] = {"status": "falta", "msg": caminho_ficheiro}
                if detalhado:
                    print(f"[CARREGAR_REDES] Ficheiro não encontrado para agente {idx}: {caminho_ficheiro}")
            except Exception as e:
                resumo[idx] = {"status": "erro", "msg": str(e)}
                if detalhado:
                    print(f"[CARREGAR_REDES] Erro ao carregar agente {idx} de {caminho_ficheiro}: {e}")

        return resumo

    def carregar_automatico_se_flag(self, padrao: str = None, mapa_ficheiros: dict = None, agentes: list = None, detalhado: bool = True):
        if getattr(self, "autoload_models", False):
            return self.carregar_redes(padrao=padrao, mapa_ficheiros=mapa_ficheiros, agentes=agentes, detalhado=detalhado)
        if detalhado:
            print("[carregar_automatico_se_flag] autoload_models não ativado; nenhum ficheiro carregado.")
        return {}


    def fase_treino(self):
        ficheiros_salvos = {}

        for idx, agente in enumerate(self.agentes):
            if not getattr(agente, "treinavel", False):
                continue

            tipo = getattr(agente, "tipo_estrategia", None)
            conf = getattr(agente, "estrategia_conf", None)
            if conf is None:
                conf = self._obter_config_estrategia(tipo or "genetica")

            acoes_disponiveis = ["cima", "baixo", "direita", "esquerda"]
            if getattr(self.ambiente, "recursos", None):
                if "recolher" not in acoes_disponiveis:
                    acoes_disponiveis.append("recolher")
            if getattr(self.ambiente, "ninhos", None):
                if "depositar" not in acoes_disponiveis:
                    acoes_disponiveis.append("depositar")

            if hasattr(agente, "set_action_space"):
                agente.set_action_space(acoes_disponiveis)
            else:
                agente.nomes_accao = acoes_disponiveis

            try:
                tamanho_entrada = agente.obter_tamanho_entrada()
            except Exception:
                alcance = getattr(agente.sensores, "alcance", 3)
                tamanho_entrada = (2 * alcance + 1) ** 2 - 1 + 5

            tamanho_saida = len(acoes_disponiveis)

            if tipo == "genetica":
                ag = EstrategiaGenetica(
                    tamanho_populacao=conf.get("populacao_tamanho", 100),
                    taxa_mutacao=conf.get("taxa_mutacao", 0.01),
                    num_geracoes=conf.get("num_ger", 25),
                    tamanho_torneio=conf.get("tamanho_torneio", 2),
                    elitismo_frac=conf.get("elitismo_frac", 0.1),
                    arq_rn=(tamanho_entrada, tamanho_saida, conf.get("hidden", (16, 8))),
                    passos_por_avaliacao=conf.get("passos_por_avaliacao", 750),
                    desvio_mutacao=conf.get("mutation_std", 0.1),
                )

                self.ambiente.posicoes = {}

                melhores_pesos, melhor_rn = ag.treinar(self.ambiente, detalhado=True, tamanho_entrada=tamanho_entrada)

                self.ambiente.reiniciar()

                agente.rede_neuronal = melhor_rn
                agente.pesos = np.array(melhores_pesos).copy()

                caminho, meta = _salvar_melhor_rn(
                    self.ambiente,
                    idx,
                    melhores_pesos,
                    melhor_rn,
                    arq_rn=getattr(ag, "arq_rn", None),
                    tipo="genetica"
                )

                meta = dict(meta or {})
                meta["tipo"] = "genetica"
                ficheiros_salvos[idx] = {"path": caminho, "meta": meta}
                print(f"[fase_treino] Agente {idx}: rede genética aplicada e salva em {caminho}")

            elif tipo == "dqn":
                conf_dqn = conf or {}
                dqn = EstrategiaQLearning(
                    arq_rn=(tamanho_entrada, tamanho_saida, conf_dqn.get("hidden", (16, 8))),
                    episodios=conf_dqn.get("episodes", 100),
                    passos_por_ep=conf_dqn.get("passos_por_ep", 750),
                    gama=conf_dqn.get("gamma", 0.99),
                    epsilon=conf_dqn.get("epsilon", 0.90),
                    epsilon_min=conf_dqn.get("epsilon_min", 0.1),
                    decaimento_epsilon=conf_dqn.get("epsilon_decay", 0.95),
                    tamanho_lote=conf_dqn.get("batch_size", 32),
                    freq_atualizacao_alvo=conf_dqn.get("target_update_freq", 50),
                    tamanho_memoria=conf_dqn.get("memory_size", 50000),
                    taxa_aprendizagem=conf_dqn.get("learning_rate", 0.001),
                )

                self.ambiente.posicoes = {}

                melhores_pesos, melhor_rn = dqn.treinar(self.ambiente, detalhado=True, alcance=agente.sensores.alcance)

                self.ambiente.reiniciar()

                agente.rede_neuronal = melhor_rn
                agente.pesos = np.array(melhores_pesos).copy()

                caminho, meta = _salvar_melhor_rn(
                    self.ambiente,
                    idx,
                    melhores_pesos,
                    melhor_rn,
                    arq_rn=getattr(dqn, "arq_rn", None),
                    tipo="dqn"
                )
                meta = dict(meta or {})
                meta["tipo"] = "dqn"
                ficheiros_salvos[idx] = {"path": caminho, "meta": meta}
                print(f"[fase_treino] Agente {idx}: rede dqn aplicada e salva em {caminho}")

            else:
                continue

        return ficheiros_salvos

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

            # Fix compatibility to check terminaton
            if hasattr(self.ambiente, "terminou"):
                todos_terminaram = self.ambiente.terminou(self.agentes)
            else:
                # Fallback
                 if isinstance(self.ambiente, AmbienteFarol):
                    todos_terminaram = all(getattr(ag, "encontrou_objetivo", False) for ag in self.agentes)
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

        obs_raw = getattr(self.ambiente, "obstaculos_brutos", None)
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

    def realizar_experiencias(self, num_execucoes: int = 30, max_passos: int = None, mapa_ficheiros: dict = None, agentes_para_carregar: list = None, semente: int = None, salvar_plot: str = None):
        import numpy as _np
        import random

        if mapa_ficheiros:
            try:
                self.carregar_redes(mapa_ficheiros=mapa_ficheiros, agentes=agentes_para_carregar or list(mapa_ficheiros.keys()))
            except Exception as e:
                print(f"[realizar_experiencias] Aviso ao carregar redes: {e}")

        visual_original = getattr(self, "visualizacao", False)
        self.visualizacao = False

        nomes_agentes = [getattr(a, "nome", f"Agente_{i}") for i, a in enumerate(self.agentes)]
        historico_fitness = []

        eh_foraging = "AmbienteForaging" in getattr(self.ambiente, "nome", "") or "Foraging" in str(type(self.ambiente))

        if eh_foraging:
            recursos_por_agente = {nome: {"recolhidos": 0, "depositados": 0} for nome in nomes_agentes}
        else:
            sucessos_por_agente = {nome: 0 for nome in nomes_agentes}

        for i in range(num_execucoes):
            if semente is not None:
                s = int(semente) + i
                _np.random.seed(s)
                random.seed(s)

            if hasattr(self.ambiente, "reiniciar"):
                self.ambiente.reiniciar()
            elif hasattr(self.ambiente, "reset"):
                self.ambiente.reset()
                
            self.ambiente.posicoes = {}

            for agente in self.agentes:
                agente.pos = None
                agente.recompensa_total = 0.0
                agente.encontrou_objetivo = False

                if hasattr(agente, "recursos_recolhidos"):
                    agente.recursos_recolhidos = 0
                if hasattr(agente, "recursos_depositados"):
                    agente.recursos_depositados = 0

            self.executa(max_passos=max_passos)

            fitness_total = 0.0
            for idx, agente in enumerate(self.agentes):
                nome = getattr(agente, "nome", f"Agente_{idx}")
                fitness_total += float(getattr(agente, "recompensa_total", 0.0) or 0.0)
                if eh_foraging:
                    recolh = int(getattr(agente, "recursos_recolhidos", 0) or 0)
                    depos = int(getattr(agente, "recursos_depositados", 0) or 0)
                    recursos_por_agente[nome]["recolhidos"] += recolh
                    recursos_por_agente[nome]["depositados"] += depos
                else:
                    if bool(getattr(agente, "encontrou_objetivo", False)):
                        sucessos_por_agente[nome] += 1

            historico_fitness.append(fitness_total)

            try:
                self.ambiente.posicoes = {}
            except Exception:
                pass

        self.visualizacao = visual_original

        array_fitness = _np.array(historico_fitness, dtype=float)
        estatisticas = {
            "media_fitness": float(array_fitness.mean()) if len(array_fitness) else 0.0,
            "desvio_fitness": float(array_fitness.std()) if len(array_fitness) else 0.0
        }

        if eh_foraging:
            estatisticas["totais_recursos"] = {nome: {"recolhidos": v["recolhidos"], "depositados": v["depositados"]} for nome, v in recursos_por_agente.items()}
        else:
            estatisticas["sucessos_por_agente"] = sucessos_por_agente

        try:
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(historico_fitness) + 1), historico_fitness, marker="o")
            plt.xlabel("Simulação")
            plt.ylabel("Fitness total")
            plt.title(f"Fitness por simulação (média={estatisticas['media_fitness']:.2f}, desvio={estatisticas['desvio_fitness']:.2f})")
            plt.grid(True)
            plt.tight_layout()
            if salvar_plot:
                plt.savefig(salvar_plot.replace(".png", "_fitness.png"), dpi=200)
            plt.show()
        except Exception as e:
            print(f"[realizar_experiencias] Erro ao desenhar fitness plot: {e}")

        try:
            plt.figure(figsize=(10, 4))
            nomes = nomes_agentes
            if eh_foraging:
                recolhidos = [recursos_por_agente[n]["recolhidos"] for n in nomes]
                depositados = [recursos_por_agente[n]["depositados"] for n in nomes]
                import numpy as _np_local
                x = _np_local.arange(len(nomes))
                width = 0.35
                plt.bar(x - width / 2, recolhidos, width, label="Recolhidos")
                plt.bar(x + width / 2, depositados, width, label="Depositados")
                plt.xticks(x, nomes, rotation=45, ha="right")
                plt.ylabel("Recursos (total em todas as simulações)")
                plt.title("Recursos recolhidos e depositados por agente (Total)")
                plt.legend()
                plt.tight_layout()
                if salvar_plot:
                    plt.savefig(salvar_plot.replace(".png", "_recursos.png"), dpi=200)
                plt.show()
            else:
                sucessos = [sucessos_por_agente[n] for n in nomes]
                plt.bar(nomes, sucessos)
                plt.xlabel("Agente")
                plt.ylabel("Nº de Sucessos")
                plt.title("Sucessos por Agente (Total)")
                plt.tight_layout()
                if salvar_plot:
                    plt.savefig(salvar_plot.replace(".png", "_sucessos.png"), dpi=200)
                plt.show()

        except Exception as e:
            print(f"[realizar_experiencias] Erro ao desenhar bar plot: {e}")

if __name__ == "__main__":
    simulador = MotorDeSimulacao().cria("simulador_foraging.json")
    if simulador.ativo:
        mapa_ficheiros = {
            2: "models/AmbienteForaging_agente2_genetica_v1.pkl",
            3: "models/AmbienteForaging_agente3_dqn_v1.pkl"
        }
        resumo = simulador.carregar_redes(mapa_ficheiros=mapa_ficheiros, agentes=[2,3])
        print(resumo)
        #resultados = simulador.realizar_experiencias(num_execucoes=30, max_passos=750, mapa_ficheiros=mapa_ficheiros, semente=20, salvar_plot="results/aggregate.png")
        #print("Resumo:", resultados["summary"])
        #simulador.fase_treino()
        simulador.fase_teste()
        #simulador.salvar_animacao_gif("models/trajetorias_foraging.gif", fps=12, trail_len=30)

