import math
import random
from copy import deepcopy

from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Ambientes.Ambiente import Ambiente
from ProjetoAA.Aprendizagem.Politicas import DIRECOES
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Objetos.Observacao import Observacao


class AmbienteForaging(Ambiente):
    def __init__(self, largura=30, altura=30, recursos=None, ninhos=None, obstaculos=None):
        super().__init__(largura, altura, recursos, obstaculos)

        self.initial_recursos = {
            tuple(r["pos"]): {"valor": r["valor"], "quantidade": r["quantidade"]}
            for r in (recursos or [])
        }
        self.recursos = deepcopy(self.initial_recursos)

        self.ninhos = [tuple(n) for n in (ninhos or [])]

        self.obstaculos = {tuple(o["pos"]) for o in (obstaculos or [])}

        self.posicoes = {}
        self.cargas = {}
        self.tempo = 0
        self.targets = {}

    def observacao_para(self, agente):
        pos = self.posicoes.get(agente, (0, 0))

        percepcoes = agente.sensores.perceber(self, pos)

        if pos in self.recursos:
            r = self.recursos[pos]
            percepcoes.append({
                "pos": pos,
                "tipo": "recurso",
                "valor": r["valor"]
            })

        if pos in self.ninhos:
            percepcoes.append({
                "pos": pos,
                "tipo": "ninho"
            })

        alcance = getattr(agente.sensores, "alcance", 3)
        for outro_agente, outra_pos in self.posicoes.items():
            if outro_agente is agente:
                continue
            ox, oy = outra_pos
            px, py = pos
            if abs(ox - px) <= alcance and abs(oy - py) <= alcance:
                percepcoes.append({
                    "pos": tuple(outra_pos),
                    "tipo": "agente",
                    "ref": outro_agente
                })

        obs = Observacao(percepcoes)
        obs.posicao = pos
        obs.carga = self.cargas.get(agente, 0)
        obs.largura = self.largura
        obs.altura = self.altura
        obs.goal = self.targets.get(agente)
        agente.ultima_obs = obs
        obs.foraging = True

        return obs

    def agir(self, accao: Accao, agente: Agente):
        pos = list(self.posicoes.get(agente, (0, 0)))
        dx, dy = DIRECOES.get(accao.nome, (0, 0))
        nova_pos = [pos[0] + dx, pos[1] + dy]

        if not (0 <= nova_pos[0] < self.largura and 0 <= nova_pos[1] < self.altura):
            return -1

        if tuple(nova_pos) in self.obstaculos:
            return -1

        current_pos = tuple(pos)
        target = self.targets.get(agente)
        carrying = self.cargas.get(agente, 0) > 0
        if target is None:
            if carrying:
                target = self._nearest_ninho(current_pos)
            else:
                target = self._nearest_resource(current_pos)
            if target is not None:
                self.targets[agente] = target

        prev_dist = self._normalized_distance(current_pos, target)

        self.posicoes[agente] = tuple(nova_pos)
        pos = nova_pos
        agente.pos = list(self.posicoes[agente])

        new_pos = tuple(pos)
        if accao.nome == "recolher" and new_pos in self.recursos:
            if self.cargas.get(agente, 0) < 1:
                recurso = self.recursos[new_pos]
                recompensa = 5
                self.cargas[agente] = 1
                recurso["quantidade"] -= 1
                if hasattr(agente, "recursos_recolhidos"):
                    agente.recursos_recolhidos += 1
                if recurso["quantidade"] <= 0:
                    del self.recursos[new_pos]
                    self._invalidate_targets_for_resource(new_pos)
                    if self.targets.get(agente) == new_pos:
                        self.targets[agente] = self._nearest_ninho(new_pos) or self._nearest_resource(new_pos)
                self.targets[agente] = self._nearest_ninho(new_pos) or self._nearest_resource(new_pos)
                return recompensa
            else:
                recompensa = 1
                return recompensa

        if accao.nome == "depositar" and new_pos in self.ninhos:
            carga = self.cargas.get(agente, 0)
            if carga > 0:
                recompensa = 10
                self.cargas[agente] = 0
                if hasattr(agente, "recursos_depositados"):
                    agente.recursos_depositados += carga
                self.targets[agente] = self._nearest_resource(new_pos)
                return recompensa
            else:
                recompensa = 1
                return recompensa

        target_after = self.targets.get(agente)
        new_dist = self._normalized_distance(new_pos, target_after)

        if target_after is None:
            recompensa = 0.1
        else:
            recompensa = -0.05
            if new_dist is not None and prev_dist is not None and new_dist < prev_dist:
                recompensa += 1.0

        return recompensa

    def atualizacao(self):
        self.tempo += 1

    def reset(self):
        self.posicoes = {}
        self.cargas = {}
        self.tempo = 0
        self.targets = {}
        for pos, info in self.recursos.items():
            info["quantidade"] = max(1, info.get("quantidade", 1))
        self.recursos = deepcopy(self.initial_recursos)

    def _nearest_resource(self, pos):
        if not self.recursos:
            return None
        px, py = pos
        best = None
        best_d = None
        for rpos in self.recursos.keys():
            dx = rpos[0] - px
            dy = rpos[1] - py
            d = math.hypot(dx, dy)
            if best_d is None or d < best_d:
                best_d = d
                best = rpos
        return best

    def _nearest_ninho(self, pos):
        if not self.ninhos:
            return None
        px, py = pos
        best = None
        best_d = None
        for npos in self.ninhos:
            dx = npos[0] - px
            dy = npos[1] - py
            d = math.hypot(dx, dy)
            if best_d is None or d < best_d:
                best_d = d
                best = npos
        return best

    def _normalized_distance(self, pos, target):
        if target is None:
            return None
        dx = pos[0] - target[0]
        dy = pos[1] - target[1]
        dist = math.hypot(dx, dy)
        max_dist = math.hypot(self.largura, self.altura) or 1.0
        return dist / max_dist

    def _invalidate_targets_for_resource(self, resource_pos):
        for ag, tgt in list(self.targets.items()):
            if tgt == resource_pos:
                agent_pos = self.posicoes.get(ag)
                if agent_pos is None:
                    self.targets[ag] = None
                else:
                    if self.cargas.get(ag, 0) > 0:
                        self.targets[ag] = self._nearest_ninho(agent_pos)
                    else:
                        self.targets[ag] = self._nearest_resource(agent_pos)


