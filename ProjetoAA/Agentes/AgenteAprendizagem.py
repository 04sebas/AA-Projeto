import math
import random

import numpy as np

from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Objetos.Sensor import Sensor


class AgenteAprendizagem(Agente):
    def __init__(self, nome="AG", politica=None, posicao=None, nomes_accao=None):
        self.neural_network = None
        self.weights = None
        if posicao is None:
            posicao = [0, 0]
        super().__init__(posicao, nome)
        politica = politica or {}
        alcance = politica.get("alcance", 1)
        self.sensores = Sensor(alcance=alcance)
        self.ultima_obs = None
        self.politica = politica
        self.trainable = True

        if nomes_accao is None:
            self.nomes_accao = ["cima", "baixo", "direita", "esquerda","recolher", "depositar"]
        else:
            self.nomes_accao = list(nomes_accao)

        self.recursos_recolhidos = 0
        self.recursos_depositados = 0
        self.tipo_estrategia = politica.get("tipo_estrategia", "genetica")
        self.estrategia_conf = politica.get("estrategia_conf", {})
        self.found_goal = False
        self.resources = set()
        self.ninhos = set()
        self.last_resource_value = 0

    def observacao(self, obs):
        self.ultima_obs = obs

        pos_atual = tuple(getattr(obs, "posicao", None))
        percepcoes = getattr(obs, "percepcoes", []) or []

        for p in percepcoes:
            tipo = p.get("tipo")
            pos_t = tuple(p.get("pos"))

            if tipo in ("recurso", "farol"):
                quantidade = p.get("quantidade", 0)
                if quantidade and quantidade > 0:
                    self.resources.add(pos_t)
                else:
                    self.resources.discard(pos_t)

            elif tipo == "ninho":
                self.ninhos.add(pos_t)

            elif tipo == "agente":
                de_agente = p.get("ref")
                if de_agente is not None:
                    self.comunica("foraging", de_agente)

        if pos_atual in self.resources:
            if not any(tuple(p.get("pos")) == pos_atual and p.get("tipo") in ("recurso", "farol") for p in percepcoes):
                self.resources.discard(pos_atual)

    def cria(self, ficheiro_parametros):
        return self

    def age(self):
        if self.neural_network is None or self.ultima_obs is None:
            return Accao(random.choice(self.nomes_accao[:4]))

        obs = self.ultima_obs
        px, py = obs.posicao
        percepcoes = obs.percepcoes or []

        if any(tuple(p.get("pos")) == (px, py) and p.get("tipo") in ("recurso", "farol") for p in percepcoes):
            if not getattr(obs, "carga", 0) > 0:
                return Accao("recolher")

        if any(tuple(p.get("pos")) == (px, py) and p.get("tipo") == "ninho" for p in percepcoes):
            if getattr(obs, "carga", 0) > 0:
                return Accao("depositar")

        if getattr(obs, "carga", 0) > 0:
            ninhos_visiveis = [tuple(p.get("pos")) for p in percepcoes if p.get("tipo") == "ninho"]
            if ninhos_visiveis:
                target = min(ninhos_visiveis, key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
                return Accao(self.__move_toward_action(target, (px, py)))

        agentes_visiveis = [p.get("ref") for p in percepcoes if p.get("tipo") == "agente" and p.get("ref") is not None]
        if agentes_visiveis:
            for de_ag in agentes_visiveis:
                self.comunica("foraging", de_ag)

        if getattr(obs, "carga", 0) == 0:
            recursos_visiveis = [tuple(p.get("pos")) for p in percepcoes if p.get("tipo") in ("recurso", "farol")]
            if recursos_visiveis:
                target = min(recursos_visiveis, key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
                return Accao(self.__move_toward_action(target, (px, py)))

        goal = getattr(obs, "goal", None)
        if goal is None:
            return Accao(random.choice(self.nomes_accao[:4]))

        nn_input = self.build_nn_input(self.ultima_obs)

        if self.tipo_estrategia in ("dqn", "genetica"):
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
        if goal is not None:
            gx, gy = goal
            dist_current = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        else:
            gx = gy = None
            dist_current = None

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
                        if getattr(obs, "carga", 0) <= 0:
                            features.append(0.9)
                        else:
                            features.append(0.1)
                    elif tipo == "ninho":
                        if getattr(obs, "carga", 0) > 0:
                            features.append(1.0)
                        else:
                            features.append(0.1)
                    else:
                        features.append(0.0)
                else:
                    if gx is not None and gy is not None:
                        dist_cell = math.sqrt((pos_check[0] - gx) ** 2 + (pos_check[1] - gy) ** 2)
                        if dist_cell < dist_current:
                            features.append(0.9)
                        else:
                            features.append(-0.9)
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
            goal_x = (gx - px) / largura
            goal_y = (gy - py) / altura
        else:
            goal_x = 0.0
            goal_y = 0.0

        carrying = float(getattr(obs, "carga", 0) > 0)

        return np.concatenate(([norm_x, norm_y], features, [goal_x, goal_y, carrying])).astype(np.float32)

    def get_input_size(self):
        alcance = self.sensores.alcance
        num_features = (2 * alcance + 1) ** 2 - 1
        return int(num_features + 5)

    def set_action_space(self, nomes_accao):
        self.nomes_accao = list(nomes_accao)

    def __move_toward_action(self, target, current):
        tx, ty = target
        cx, cy = current
        dx = tx - cx
        dy = ty - cy

        if abs(dx) > abs(dy):
            if dx > 0:
                return "direita"
            elif dx < 0:
                return "esquerda"
        else:
            if dy > 0:
                return "baixo"
            elif dy < 0:
                return "cima"

        return "ficar"




