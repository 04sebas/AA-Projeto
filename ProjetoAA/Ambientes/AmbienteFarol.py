import math
from ProjetoAA.Ambientes.Ambiente import Ambiente
from ProjetoAA.Objetos.Observacao import Observacao

class AmbienteFarol(Ambiente):
    def __init__(self, largura=100, altura=100, pos_farol=(50,75), obstaculos=None):
        self.largura = int(largura)
        self.altura = int(altura)
        self.pos_farol = tuple(pos_farol)

        self.obstaculos = set()
        self.obstaculos_raw = []
        for o in (obstaculos or []):
            pos = o.get("pos") if isinstance(o, dict) else None
            if pos is None:
                continue
            tup = (int(pos[0]), int(pos[1]))
            self.obstaculos.add(tup)
            self.obstaculos_raw.append({"pos": [tup[0], tup[1]]})

        self.recursos = {tuple(self.pos_farol): {"tipo": "farol", "pos": list(self.pos_farol), "valor": 1500, "quantidade": 1}}

        super().__init__(
            largura=self.largura,
            altura=self.altura,
            recursos=self.recursos,
            obstaculos=self.obstaculos
        )

        self.posicoes = {}
        self.tempo = 0

    def observacao_para(self, agente):
        pos = tuple(self.posicoes.get(agente, (0,0)))
        percepcoes = agente.sensores.perceber(self, pos)

        obs = Observacao(percepcoes)
        obs.posicao = pos
        obs.largura = self.largura
        obs.altura = self.altura
        agente.ultima_obs = obs
        obs.goal = self.pos_farol
        obs.foraging = False

        return obs

    def agir(self, accao, agente):
        pos = list(self.posicoes.get(agente, (0, 0)))
        x, y = pos

        if accao.nome == "recolher" or (x, y) == self.pos_farol:
            recurso = self.recursos.get((x, y))
            if recurso is not None:
                valor = recurso.get("valor", 1500)
                agente.found_goal = True
                return valor
            return -0.1
        elif accao.nome == "cima":
            dx, dy = (0, 1)
        elif accao.nome == "baixo":
            dx, dy = (0, -1)
        elif accao.nome == "esquerda":
            dx, dy = (-1, 0)
        elif accao.nome == "direita":
            dx, dy = (1, 0)
        else:
            dx, dy = (0, 0)

        newx, newy = x + dx, y + dy



        if not (0 <= newx < self.largura and 0 <= newy < self.altura):
            return -1

        if (newx, newy) in self.obstaculos:
            return -1

        prev_dist = self.distance_to_goal(x, y)

        self.posicoes[agente] = (newx, newy)
        agente.pos = (newx, newy)

        new_dist = self.distance_to_goal(newx, newy)

        reward = -0.05
        if new_dist < prev_dist:
            reward += 1.0

        return reward

    def atualizacao(self):
        self.tempo += 1

    def reset(self):
        self.posicoes = {}
        self.tempo = 0

    def distance_to_goal(self, x, y):
        dx = x - self.pos_farol[0]
        dy = y - self.pos_farol[1]
        dist = math.sqrt(dx * dx + dy * dy)
        max_dist = math.sqrt(self.largura * self.largura + self.altura * self.altura)
        return dist / max_dist



