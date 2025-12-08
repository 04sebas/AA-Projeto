import random
from Agentes.Agente import Agente
from Aprendizagem.Politicas import DIRECOES
from Objetos.Accao import Accao
from Objetos.Observacao import Observacao
from Ambientes.Ambiente import Ambiente


class AmbienteForaging(Ambiente):
    def __init__(self, largura=30, altura=30, recursos=None, ninhos=None, obstaculos=None):
        super().__init__(largura, altura, recursos, obstaculos)

        self.recursos = {
            tuple(r["pos"]): {"valor": r["valor"], "quantidade": r["quantidade"]}
            for r in (recursos or [])
        }

        self.ninhos = [tuple(n) for n in (ninhos or [])]

        self.obstaculos = {tuple(o["pos"]) for o in (obstaculos or [])}

        self.posicoes = {}
        self.cargas = {}
        self.tempo = 0

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

        obs = Observacao(percepcoes)

        obs.posicao = pos
        obs.carga = self.cargas.get(agente, 0)

        obs.largura = self.largura
        obs.altura = self.altura

        agente.ultima_obs = obs

        return obs

    def agir(self, accao: Accao, agente: Agente):
        pos = list(self.posicoes.get(agente, (0, 0)))
        dx, dy = DIRECOES.get(accao.nome, (0, 0))
        nova_pos = [pos[0] + dx, pos[1] + dy]

        if 0 <= nova_pos[0] < self.largura and 0 <= nova_pos[1] < self.altura:
            if tuple(nova_pos) not in self.obstaculos:
                self.posicoes[agente] = tuple(nova_pos)
                pos = nova_pos

        recompensa = 0
        pos_tuple = tuple(pos)

        if accao.nome == "recolher" and pos_tuple in self.recursos:
            if self.cargas.get(agente, 0) < 1:
                recurso = self.recursos[pos_tuple]
                recompensa = recurso["valor"]
                self.cargas[agente] = 1
                recurso["quantidade"] -= 1
                if hasattr(agente, "recursos_recolhidos"):
                    agente.recursos_recolhidos += 1
                if recurso["quantidade"] <= 0:
                    del self.recursos[pos_tuple]

        if accao.nome == "depositar" and pos_tuple in self.ninhos:
            carga = self.cargas.get(agente, 0)
            recompensa = carga * 10
            self.cargas[agente] = 0
            if hasattr(agente, "recursos_depositados"):
                agente.recursos_depositados += carga

        agente.pos = list(self.posicoes[agente])
        return recompensa

    def atualizacao(self):
        self.tempo += 1

    def posicao_aleatoria(self):
        while True:
            x = random.randint(0, self.largura - 1)
            y = random.randint(0, self.altura - 1)
            if (x, y) not in self.obstaculos:
                return x, y
