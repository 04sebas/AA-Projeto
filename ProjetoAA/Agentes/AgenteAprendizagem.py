import math

import numpy as np

from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Objetos.Sensor import Sensor


class AgenteAprendizagem(Agente):
    def __init__(self, nome="AG", politica=None, posicao=None):
        self.neural_network = None
        self.weights = None
        if posicao is None:
            posicao = [0, 0]
        super().__init__(posicao, nome)
        politica = politica or {}
        alcance = politica.get("alcance", 3)
        self.sensores = Sensor(alcance=alcance)
        self.ultima_obs = None
        self.politica = politica
        self.trainable = True
        self.nomes_accao = ["cima","baixo","direita","esquerda"]
        self.recursos_recolhidos = 0
        self.recursos_depositados = 0
        self.tipo_estrategia = politica.get("tipo_estrategia", "genetica")
        self.estrategia_conf = politica.get("estrategia_conf", {})
        self.found_goal = False
        self.resources = set()
        self.ninhos = set()


    def observacao(self, obs):
        self.ultima_obs = obs

        pos_atual = tuple(getattr(obs, "posicao", None))
        percepcoes = getattr(obs, "percepcoes", []) or []

        for p in percepcoes:
            tipo = p.get("tipo")
            pos_t = tuple(p.get("pos"))

            if tipo in ("recurso", "farol"):
                quantidade = p.get("quantidade", p.get("valor", 1))
                if quantidade and quantidade > 0:
                    self.resources.add(pos_t)
                else:
                    self.resources.discard(pos_t)

            elif tipo == "ninho":
                self.ninhos.add(pos_t)

        if pos_atual in self.resources:
            if not any(tuple(p.get("pos")) == pos_atual and p.get("tipo") in ("recurso", "farol")
                       for p in percepcoes):
                self.resources.discard(pos_atual)

    def cria(self, ficheiro_parametros):
        return self

    def age(self):
        if self.neural_network is None or self.ultima_obs is None:
            return Accao("ficar")

        nn_input = self.build_nn_input(self.ultima_obs)

        if self.tipo_estrategia == "dqn" or self.tipo_estrategia == "genetica":
            output = self.neural_network.forward(nn_input)
            action_index = int(np.argmax(output))
        else:
            return Accao("ficar")

        action_index = max(0, min(action_index, len(self.nomes_accao) - 1))
        return Accao(self.nomes_accao[action_index])

    def avaliacao_estado_atual(self, recompensa):
        self.recompensa_total += recompensa

    def comunica(self, mensagem, de_agente):
        if mensagem == "farol":
            return

        if mensagem == "foraging":
            recursos_outro = getattr(de_agente, "resources", None)
            entregas_outro = getattr(de_agente, "ninhos", None)

            if isinstance(recursos_outro, set):
                self.resources.update(recursos_outro)

            if isinstance(entregas_outro, set):
                self.ninhos.update(entregas_outro)

            return

    def surroundings(self):
        obs = self.ultima_obs
        if not obs:
            input_size = (2 * self.sensores.alcance + 1) ** 2 - 1
            return [-0.9] * input_size

        px, py = obs.posicao
        alcance = self.sensores.alcance
        percepcoes = obs.percepcoes or []
        features = []

        goal = getattr(obs, "goal", None)
        gx, gy = goal if goal else (None, None)

        for dy in range(-alcance, alcance + 1):
            for dx in range(-alcance, alcance + 1):
                if dx == 0 and dy == 0:
                    continue

                pos_check = (px + dx, py + dy)
                objeto = next((p for p in percepcoes if tuple(p["pos"]) == pos_check), None)

                if objeto:
                    tipo = objeto.get("tipo", "")
                    if tipo == "obstaculo":
                        features.append(-0.9)
                    elif tipo in ("recurso", "farol"):
                        if getattr(obs, "foraging", False) and not getattr(obs, "carga", 0) > 0:
                            features.append(1.0)
                        else:
                            features.append(0.9)
                    elif tipo == "ninho":
                        if getattr(obs, "carga", 0) > 0:
                            features.append(1.0)
                        else:
                            features.append(0.1)
                    else:
                        features.append(-0.1)
                else:
                    if gx is not None and gy is not None:
                        dx_goal = (gx - pos_check[0]) / max(1, obs.largura - 1)
                        dy_goal = (gy - pos_check[1]) / max(1, obs.altura - 1)
                        features.append((dx_goal + dy_goal) / 2.0)
                    else:
                        features.append(-0.9)

        return features

    def build_nn_input(self, obs):
        px, py = obs.posicao
        features = np.array(self.surroundings(), dtype=np.float32)
        largura = max(1.0, getattr(obs, "largura", 1))
        altura = max(1.0, getattr(obs, "altura", 1))
        norm_x = px / largura
        norm_y = py / altura
        goal = getattr(obs, "goal", None)
        if goal is not None:
            gx, gy = goal
            max_dist = math.sqrt(obs.largura ** 2 + obs.altura ** 2) or 1.0
            dist_norm = 1.0 - math.sqrt((px - gx) ** 2 + (py - gy) ** 2) / max_dist
            dist_norm = float(np.clip(dist_norm, 0.0, 1.0))
        else:
            dist_norm = 0.0
        return np.concatenate(([norm_x, norm_y], features, [dist_norm])).astype(np.float32)

    def get_input_size(self):
        alcance = getattr(self.sensores, "alcance", 3)
        num_features = (2 * alcance + 1) ** 2 - 1
        return int(num_features + 3)



