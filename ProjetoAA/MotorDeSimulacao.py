import json
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from Ambientes.AmbienteForaging import AmbienteForaging
from Agentes.AgenteAprendizagem import AgenteAprendizagem
from Agentes.AgenteFixo import AgenteFixo
from Ambientes.AmbienteFarol import AmbienteFarol

class MotorDeSimulacao:
    def __init__(self):
        self.recursos_iniciais = None
        self.ambiente = None
        self.passos = 0
        self.agentes = []
        self.order = []
        self.ativo = False
        self.max_passos = 100
        self.visualizacao = True
        self.historico_agentes = {}
        self.historico_recompensa = {agente: [0] for agente in self.agentes}

    def lista_agentes(self):
        return self.agentes

    def cria(self, nome_do_ficheiro_parametros: str) -> "MotorDeSimulacao":
        try:
            with open(nome_do_ficheiro_parametros, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._cria_ambiente(config.get("ambiente", {}))
            self._cria_agentes(config.get("agentes", []))
            simulador_config = config.get("simulador", {})
            self.max_passos = simulador_config.get("max_passos", 100)
            self.visualizacao = simulador_config.get("visualizacao", True)
            self.ativo = True
            print(f"Simulação criada com sucesso: {len(self.agentes)} agentes no ambiente")
        except FileNotFoundError:
            print(f"Erro: Ficheiro {nome_do_ficheiro_parametros} não encontrado.")
        except json.JSONDecodeError:
            print(f"Erro: Ficheiro JSON inválido: {nome_do_ficheiro_parametros}")
        except Exception as e:
            print(f"Erro a criar simulação. {e}")
        return self

    def _cria_ambiente(self, config_ambiente: dict):
        tipo_ambiente = config_ambiente.get("tipo", "AmbienteFarol")
        if tipo_ambiente == "AmbienteFarol":
            self.ambiente = AmbienteFarol(
                largura=config_ambiente.get("largura", 10),
                altura=config_ambiente.get("altura", 10),
                pos_farol=tuple(config_ambiente.get("pos_farol", [5, 5])),
                obstaculos=config_ambiente.get("obstaculos", [])
            )
        elif tipo_ambiente == "AmbienteForaging":
            if AmbienteForaging is None:
                raise ImportError("AmbienteForaging não está disponível — certifica-te que o ficheiro existe e o import funciona.")
            recursos = config_ambiente.get("recursos", [])
            ninhos = [tuple(n) for n in config_ambiente.get("ninhos", [])]
            obstaculos = config_ambiente.get("obstaculos", [])
            self.ambiente = AmbienteForaging(
                largura=config_ambiente.get("largura", 30),
                altura=config_ambiente.get("altura", 30),
                recursos=recursos,
                ninhos=ninhos,
                obstaculos=obstaculos
            )
        else:
            raise ValueError(f"Tipo de ambiente não suportado: {tipo_ambiente}")

    def _cria_agentes(self, config_agentes):
        for config in config_agentes:
            tipo = config.get("tipo")
            quantidade = config.get("quantidade", 1)
            posicoes = config.get("posicao_inicial", "random")
            tipo_estrategia = config.get("tipo_estrategia")
            sensores = config.get("sensores", 1)

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
                    agente = AgenteFixo(posicao=list(pos), politica=config.get("politica"))
                elif tipo == "AgenteAprendizagem":
                    agente_prototipo = AgenteAprendizagem(tipo_estrategia=tipo_estrategia, sensores=sensores)
                    agente = agente_prototipo.cria(config.get("ficheiro_parametros", "simulacao_farol.json"))
                else:
                    raise ValueError(f"Tipo de agente desconhecido: {tipo}")

                agente.pos = list(pos)
                self.ambiente.posicoes[agente] = tuple(pos)

                self.agentes.append(agente)

    def fase_treino(self):
        for agente in self.agentes:
            if isinstance(agente, AgenteAprendizagem):
                if agente.tipo_estrategia == "genetica":
                    agente.estrategia.run(self.ambiente)

    def fase_teste(self):
        self.executa(self.max_passos)

    def executa(self, max_passos: int = None):
        if not self.ativo:
            print("Simulação não foi criada corretamente.")
            return
        max_p = max_passos if max_passos is not None else self.max_passos
        for agente in self.agentes:
            self.historico_agentes[agente] = [list(agente.pos)]
            self.historico_recompensa[agente] = [0]

        self.recursos_iniciais = {}
        if isinstance(self.ambiente.recursos, dict):
            for pos, info in self.ambiente.recursos.items():
                self.recursos_iniciais[pos] = {
                    "valor": info["valor"],
                    "quantidade": info["quantidade"]
                }

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
                ultimo_valor = self.historico_recompensa[agente][-1]
                self.historico_recompensa[agente].append(ultimo_valor + recompensa)

            self.ambiente.atualizacao()
            self.passos += 1

            if isinstance(self.ambiente, AmbienteFarol):
                todos_terminaram = all(
                    tuple(agente.pos) == self.ambiente.pos_farol for agente in self.agentes
                )
            elif isinstance(self.ambiente, AmbienteForaging):
                todos_terminaram = len(self.ambiente.recursos) == 0 and all(
                    self.ambiente.cargas.get(agente, 0) == 0 for agente in self.agentes
                )

            if todos_terminaram:
                print(f"Simulação terminada no passo {passo}!")
                break

            self.ambiente.atualizacao()
            self.passos += 1

        if self.visualizacao:
            try:
                self.visualizar_caminhos()
            except Exception as e:
                print(f"Erro ao desenhar visualizações: {e}")

    def visualizar_heatmap(self):
        altura = self.ambiente.altura
        largura = self.ambiente.largura
        heatmap = np.zeros((altura, largura))
        for path in self.historico_agentes.values():
            for (x, y) in path:
                if 0 <= x < largura and 0 <= y < altura:
                    heatmap[y][x] += 1
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(heatmap, cmap="YlGnBu", cbar=True)
        if hasattr(self.ambiente, "pos_farol"):
            farol_x, farol_y = self.ambiente.pos_farol
            plt.text(farol_x + 0.5, farol_y + 0.5, "Farol", color='red', fontsize=10, ha='center', va='center', fontweight='bold')

        recursos = getattr(self.ambiente, "recursos", None)
        if recursos:
            if isinstance(recursos, dict):
                itens = recursos.items()
            else:
                itens = []
                for r in recursos:
                    pos = tuple(r.get("pos")) if isinstance(r.get("pos"), (list, tuple)) else None
                    if pos is not None:
                        itens.append((pos, r))
            for pos, info in itens:
                x, y = pos
                qtd = info.get("quantidade", info.get("qtd", 1))
                ax.scatter([x + 0.5], [y + 0.5], marker='o', s=50 + qtd * 10)
                ax.text(x + 0.5, y + 0.5, str(info.get("valor", qtd)), color='black', fontsize=8, ha='center', va='center')

        ninhos = getattr(self.ambiente, "ninhos", None)
        if ninhos:
            for n in ninhos:
                x, y = n
                ax.scatter([x + 0.5], [y + 0.5], marker='s', s=120, facecolors='none', edgecolors='magenta', linewidths=2)
                ax.text(x + 0.5, y + 0.5, "Ninho", color='magenta', fontsize=8, ha='center', va='center')

        obs_raw = getattr(self.ambiente, "obstaculos_raw", None)
        if obs_raw:
            obs_x = [p["pos"][0] + 0.5 for p in obs_raw]
            obs_y = [p["pos"][1] + 0.5 for p in obs_raw]
            ax.scatter(obs_x, obs_y, color="black", marker="s", s=80)
        else:
            obs_set = getattr(self.ambiente, "obstaculos", None)
            if obs_set:
                obs_x = [p[0] + 0.5 for p in obs_set]
                obs_y = [p[1] + 0.5 for p in obs_set]
                ax.scatter(obs_x, obs_y, color="black", marker="s", s=80)

        ax.invert_yaxis()
        plt.title("Heatmap das Posições Visitadas pelos Agentes")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def visualizar_caminhos(self):
        altura = self.ambiente.altura
        largura = self.ambiente.largura
        plt.figure(figsize=(8, 8))

        cores = plt.cm.tab20(np.linspace(0, 1, max(1, len(self.historico_agentes))))

        for i, (agente, path) in enumerate(self.historico_agentes.items()):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            plt.plot(xs, ys, marker='.', linewidth=1, color=cores[i], label=f"Agente {i}")
            plt.scatter(xs[0], ys[0], c=[cores[i]], marker='o', s=80)
            plt.scatter(xs[-1], ys[-1], c=[cores[i]], marker='x', s=80)

        if hasattr(self.ambiente, "pos_farol"):
            farol_x, farol_y = self.ambiente.pos_farol
            plt.scatter([farol_x], [farol_y], color="red", marker="*", s=300, label="Farol")

        recursos = getattr(self.ambiente, "recursos", None)
        if recursos:
            if isinstance(recursos, dict):
                itens = recursos.items()
            else:
                itens = []
                for r in recursos:
                    pos = tuple(r.get("pos")) if isinstance(r.get("pos"), (list, tuple)) else None
                    if pos is not None:
                        itens.append((pos, r))
            for pos, info in itens:
                x, y = pos
                plt.scatter([x], [y], s=80, marker='o', label="_nolegend_", alpha=0.7)
                plt.text(x + 0.1, y + 0.1, str(info.get("valor", info.get("quantidade", 1))), fontsize=8)

        ninhos = getattr(self.ambiente, "ninhos", None)
        if ninhos:
            for n in ninhos:
                x, y = n
                plt.scatter([x], [y], marker='s', s=140, facecolors='none', edgecolors='magenta', linewidths=2)
                plt.text(x + 0.1, y + 0.1, "Ninho", color='magenta', fontsize=8)

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
            plt.scatter(obs_x, obs_y, color="black", marker="s", s=80, label="Obstáculos")
        if hasattr(self, "recursos_iniciais"):
            for pos, info in self.recursos_iniciais.items():
                x, y = pos

                atual = self.ambiente.recursos.get(pos, {}).get("quantidade", 0)

                cor = "blue" if atual > 0 else "gray"

                plt.scatter([x], [y], s=80, marker='o', color=cor)
                plt.text(x + 0.1, y + 0.1,
                         f"ini:{info['quantidade']}\nact:{atual}",
                         fontsize=7)

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
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.show()

    def visualizar_recursos_agentes(self):
        agentes = self.agentes
        nomes = [f"Agente {i}" for i in range(len(agentes))]
        recolhidos = [getattr(a, "recursos_recolhidos", 0) for a in agentes]
        depositados = [getattr(a, "recursos_depositados", 0) for a in agentes]

        x = np.arange(len(agentes))
        largura = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - largura / 2, recolhidos, largura, label='Recolhidos', color='skyblue')
        plt.bar(x + largura / 2, depositados, largura, label='Depositados', color='lightgreen')

        plt.xlabel("Agentes")
        plt.ylabel("Quantidade de Recursos")
        plt.title("Recursos Recolhidos e Depositados por Agente")
        plt.xticks(x, nomes)
        plt.legend()
        plt.grid(axis='y')
        plt.show()

    def visualizar_recompensa_agentes(self):
        plt.figure(figsize=(10, 6))
        cores = plt.cm.tab20(np.linspace(0, 1, len(self.agentes)))
        for i, agente in enumerate(self.agentes):
            plt.plot(self.historico_recompensa[agente], label=f"Agente {i}", color=cores[i])
        plt.xlabel("Passos")
        plt.ylabel("Recompensa Acumulada")
        plt.title("Evolução da Recompensa Acumulada por Agente")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    simulador = MotorDeSimulacao().cria("simulacao_foraging.json")
    if simulador.ativo:
        simulador.fase_treino()
        simulador.fase_teste()
        simulador.visualizar_recursos_agentes()
        simulador.visualizar_recompensa_agentes()


