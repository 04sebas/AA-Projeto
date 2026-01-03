from abc import ABC, abstractmethod

class EstrategiaAprendizagem(ABC):
    def __init__(self, arq_rn=(15, 4, (16, 8)), detalhado=True):
        self.arq_rn = list(arq_rn)
        self.detalhado = detalhado
        self.historico_fitness = []
        self.historico_caminhos = []
        self.melhor_rn = None
        self.melhores_pesos = None

    @abstractmethod
    def escolher_acao(self, estado, acoes_possiveis):
        pass

    @abstractmethod
    def treinar(self, ambiente):
        """
        Método principal de treino. Deve popular self.historico_fitness e self.historico_caminhos.
        Retorna (melhores_pesos, melhor_rn).
        """
        pass

    def gerar_graficos(self, ambiente, titulo_fitness="Aptidao", titulo_caminhos="Caminhos", outros_plots=None):
        if not self.detalhado:
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import matplotlib.cm as cm
            import numpy as np

            larg = getattr(ambiente, "largura", 50)
            alt = getattr(ambiente, "altura", 50)

            recursos_brutos = getattr(ambiente, "recursos", {})
            ninhos_brutos = getattr(ambiente, "ninhos", [])
            obstaculos_brutos = getattr(ambiente, "obstaculos", [])

            recursos_lista = []
            if isinstance(recursos_brutos, dict):
                for pos, info in recursos_brutos.items():
                    recursos_lista.append((tuple(pos), dict(info)))
            elif isinstance(recursos_brutos, (list, tuple)):
                for r in recursos_brutos:
                    if isinstance(r, dict) and "pos" in r:
                        recursos_lista.append(
                            (tuple(r["pos"]), {"quantidade": r.get("quantidade", 1), "valor": r.get("valor", 0)}))

            ninhos_lista = []
            if isinstance(ninhos_brutos, (list, tuple, set)):
                for n in ninhos_brutos:
                    try:
                        ninhos_lista.append(tuple(n))
                    except:
                        pass

            obstaculos_lista = []
            if isinstance(obstaculos_brutos, (list, tuple, set)):
                for o in obstaculos_brutos:
                    if isinstance(o, tuple) and len(o) >= 2:
                        obstaculos_lista.append(tuple(o))
                    elif isinstance(o, list) and len(o) >= 2:
                        obstaculos_lista.append(tuple(o))
                    elif isinstance(o, dict) and "pos" in o:
                        obstaculos_lista.append(tuple(o["pos"]))

            # 1. Gráfico de Fitness
            if self.historico_fitness:
                plt.figure(figsize=(10, 4.5))
                plt.plot(range(len(self.historico_fitness)), self.historico_fitness, marker='o')
                plt.title(titulo_fitness)
                plt.xlabel("Iteração / Geração")
                plt.ylabel("Aptidão / Recompensa")
                plt.grid(True)
                plt.tight_layout()

            # 2. Gráfico de Caminhos (Top 5)
            if self.historico_caminhos:
                fig, ax = plt.subplots(figsize=(10, 10))

                # Desenhar Ambiente
                for (rx, ry), info in recursos_lista:
                    ax.add_patch(
                        patches.Circle((rx, ry), radius=0.4, facecolor="gold", alpha=0.6, edgecolor='k', linewidth=0.3))
                    q = info.get("quantidade", "")
                    ax.text(rx, ry, f"{q}", color="black", ha="center", va="center", fontsize=7)

                for nx, ny in ninhos_lista:
                    ax.add_patch(patches.Circle((nx, ny), radius=0.5, facecolor="blue", edgecolor='k'))
                    ax.text(nx, ny, "N", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

                for ox, oy in obstaculos_lista:
                    ax.add_patch(patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black"))

                # Plotar caminhos
                qtd_total = len(self.historico_caminhos)
                top_n = min(5, qtd_total)
                
                indices = list(range(qtd_total))
                if len(self.historico_fitness) == qtd_total:
                    # assumindo que maior fitness é melhor
                    indices_ordenados = sorted(indices, key=lambda i: self.historico_fitness[i], reverse=True)
                    melhores_indices = indices_ordenados[:top_n]
                else:
                    melhores_indices = indices[-top_n:]

                colors = cm.rainbow(np.linspace(0, 1, len(melhores_indices)))

                for idx_c, idx_real in enumerate(melhores_indices):
                    path = self.historico_caminhos[idx_real]
                    if not path: continue
                    xs = [p[0] for p in path]
                    ys = [p[1] for p in path]
                    fit_val = self.historico_fitness[idx_real] if idx_real < len(self.historico_fitness) else 0.0
                    ax.plot(xs, ys, label=f"Iter {idx_real} (Fit: {fit_val:.2f})", alpha=0.7, color=colors[idx_c])
                    ax.plot(xs[-1], ys[-1], 'x', markersize=8, color=colors[idx_c])

                ax.set_xlim(-1, larg + 1)
                ax.set_ylim(-1, alt + 1)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(titulo_caminhos)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.grid(True)
                if top_n > 0:
                    ax.legend(loc='upper right', fontsize='small')
                plt.tight_layout()

            if outros_plots:
                outros_plots(plt)

            plt.show()

        except Exception as e:
            print(f"[EstrategiaAprendizagem] Erro ao gerar gráficos: {e}")
