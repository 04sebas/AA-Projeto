from ProjetoAA.Ambientes.Ambiente import Ambiente
from ProjetoAA.Objetos.Observacao import Observacao
from ProjetoAA.Aprendizagem.Politicas import DIRECOES

class AmbienteFarol(Ambiente):
    def __init__(self, largura=100, altura=100, pos_farol=(50,75), obstaculos=None):
        super().__init__(largura, altura, obstaculos=obstaculos, nome="AmbienteFarol")
        self.pos_farol = (int(pos_farol[0]), int(pos_farol[1]))

        if self.pos_farol in self.obstaculos:
            self.obstaculos.discard(self.pos_farol)

        self.recursos = {self.pos_farol: {"tipo": "farol", "pos": list(self.pos_farol), "valor": 1500, "quantidade": 1}}
        self.obstaculos_brutos = [{"pos": [p[0], p[1]]} for p in self.obstaculos]
        self.alvos = {}

    def observacao_para(self, agente):
        pos = tuple(self.posicoes.get(agente, (0, 0)))
        percepcoes = agente.sensores.perceber(self, pos) or []

        if pos == self.pos_farol:
            percepcoes.append({
                "tipo": "farol",
                "pos": pos
            })

        obs = Observacao(percepcoes)
        obs.posicao = pos
        obs.largura = self.largura
        obs.altura = self.altura
        agente.ultima_obs = obs
        obs.carga = 0
        obs.goal = self.pos_farol
        obs.foraging = False
        return obs

    def agir(self, accao, agente):
        if getattr(agente, "encontrou_objetivo", False):
            return 0.0
        pos_bruta = self.posicoes.get(agente, (0, 0))
        x, y = int(pos_bruta[0]), int(pos_bruta[1])

        if (x, y) == self.pos_farol:
            recurso = self.recursos.get((x, y))
            if recurso:
                agente.encontrou_objetivo = True
                return float(recurso.get("valor", 1500))

        if accao.nome == "recolher":
            return -0.5

        dx, dy = DIRECOES.get(accao.nome, (0, 0))
        novax, novay = x + int(dx), y + int(dy)

        if not self.posicao_valida(novax, novay):
            return -1

        dist_antiga = self.distancia_para_objetivo(x, y)
        self.posicoes[agente] = (novax, novay)
        agente.pos = (novax, novay)
        nova_dist = self.distancia_para_objetivo(novax, novay)

        recompensa = -0.1
        if (novax, novay) == (x, y):
            recompensa -= 0.2

        if nova_dist is not None and dist_antiga is not None and nova_dist < dist_antiga:
            recompensa += 1.0

        return recompensa

    def distancia_para_objetivo(self, x, y):
        dist = abs(int(x) - self.pos_farol[0]) + abs(int(y) - self.pos_farol[1])
        max_dist = float(self.largura + self.altura) or 1.0
        return dist / max_dist

    def terminou(self, agentes=None):
        return all(getattr(a, "encontrou_objetivo", False) for a in agentes)
