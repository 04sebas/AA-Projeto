import random
from Agentes.Agente import Agente
from Ambientes.Ambiente import Ambiente
from Objetos.Accao import Accao
from Objetos.Observacao import Observacao

class AmbienteFarol(Ambiente):
    def __init__(self, largura=100, altura=100, pos_farol=(50, 75), obstaculos=None):
        super().__init__(
            largura,
            altura,
            recursos={tuple(r["pos"]): r for r in [{"tipo": "farol", "pos": pos_farol, "valor": 1500, "quantidade": 1}]},
            obstaculos=obstaculos,
        )
        self.pos_farol = pos_farol
        self.obstaculos = set()
        self.obstaculos_raw = []
        for o in (obstaculos or []):
            pos = o.get("pos")
            if pos is None:
                continue
            tup = (int(pos[0]), int(pos[1]))
            self.obstaculos.add(tup)
            self.obstaculos_raw.append({"pos": [tup[0], tup[1]]})
        self.posicoes = {}
        self.tempo = 0

    def observacao_para(self, agente):
        pos = self.posicoes.get(agente, (0, 0))
        percepcoes = []

        if hasattr(agente, "sensores"):
            percepcoes = agente.sensores.perceber(self, pos)

        obs = Observacao(percepcoes)
        obs.posicao = pos
        obs.largura = self.largura
        obs.altura = self.altura

        agente.ultima_obs = obs
        return obs

    def agir(self, accao: Accao, agente: Agente):
        pos = list(self.posicoes.get(agente, (0, 0)))

        if tuple(pos) == self.pos_farol:
            self.posicoes[agente] = tuple(pos)
            agente.pos = tuple(pos)
            return 0

        nova_pos = pos.copy()

        if accao.nome == "cima":
            nova_pos[1] -= 1
        elif accao.nome == "baixo":
            nova_pos[1] += 1
        elif accao.nome == "esquerda":
            nova_pos[0] -= 1
        elif accao.nome == "direita":
            nova_pos[0] += 1
        elif accao.nome == "ficar":
            nova_pos = pos

        distancia_atual = abs(pos[0] - self.pos_farol[0]) + abs(pos[1] - self.pos_farol[1])
        distancia_nova = abs(nova_pos[0] - self.pos_farol[0]) + abs(nova_pos[1] - self.pos_farol[1])

        if not (0 <= nova_pos[0] < self.largura) or not (0 <= nova_pos[1] < self.altura):
            nova_pos = pos
            recompensa = -1
        elif tuple(nova_pos) in self.obstaculos:
            nova_pos = pos
            recompensa = -1
        elif tuple(nova_pos) == self.pos_farol:
            valor = self.recursos.get(tuple(nova_pos), {}).get("valor", 1500)
            self.posicoes[agente] = tuple(nova_pos)
            agente.pos = tuple(nova_pos)
            return valor
        else:
            if distancia_nova < distancia_atual:
                recompensa = 2
            elif distancia_nova > distancia_atual:
                recompensa = 0.1
            else:
                recompensa = 0

        self.posicoes[agente] = tuple(nova_pos)
        agente.pos = tuple(nova_pos)

        return recompensa

    def posicao_aleatoria(self):
        x = random.randint(0, self.largura - 1)
        y = random.randint(0, self.altura - 1)
        return [x, y]

    def atualizacao(self):
        self.tempo += 1
