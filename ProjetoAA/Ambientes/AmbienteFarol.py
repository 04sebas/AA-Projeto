from ProjetoAA.Ambientes.Ambiente import Ambiente
from ProjetoAA.Objetos.Observacao import Observacao
from ProjetoAA.Aprendizagem.Politicas import DIRECOES

class AmbienteFarol(Ambiente):
    def __init__(self, largura=100, altura=100, pos_farol=(50,75), obstaculos=None):
        self.largura = int(largura)
        self.altura = int(altura)
        self.pos_farol = (int(pos_farol[0]), int(pos_farol[1]))

        parsed_obst = set()
        for o in (obstaculos or []):
            if isinstance(o, dict) and "pos" in o:
                parsed_obst.add((int(o["pos"][0]), int(o["pos"][1])))
            elif isinstance(o, (list, tuple)) and len(o) >= 2:
                parsed_obst.add((int(o[0]), int(o[1])))
        if self.pos_farol in parsed_obst:
            parsed_obst.discard(self.pos_farol)

        self.obstaculos = parsed_obst
        self.obstaculos_raw = [{"pos": [p[0], p[1]]} for p in self.obstaculos]

        self.recursos = {self.pos_farol: {"tipo": "farol", "pos": list(self.pos_farol), "valor": 1500, "quantidade": 1}}

        super().__init__(
            largura=self.largura,
            altura=self.altura,
            recursos=self.recursos,
            obstaculos=self.obstaculos,
            nome="AmbienteFarol"
        )

        self.posicoes = {}
        self.targets = {}
        self.tempo = 0

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
        if getattr(agente, "found_goal", False):
            return 0.0
        raw_pos = self.posicoes.get(agente, (0, 0))
        x, y = int(raw_pos[0]), int(raw_pos[1])

        if (x, y) == self.pos_farol:
            recurso = self.recursos.get((x, y))
            if recurso is not None:
                valor = float(recurso.get("valor", 1500))
                agente.found_goal = True
                return valor

        if accao.nome == "recolher":
            return -0.5

        dx, dy = DIRECOES.get(accao.nome, (0, 0))
        newx, newy = x + int(dx), y + int(dy)

        if not (0 <= newx < self.largura and 0 <= newy < self.altura):
            return -1
        if (newx, newy) in self.obstaculos:
            return -1

        prev_dist = self.distance_to_goal(x, y)

        self.posicoes[agente] = (newx, newy)
        agente.pos = (newx, newy)

        new_dist = self.distance_to_goal(newx, newy)

        reward = -0.1
        if (newx, newy) == (x, y):
            reward -= 0.2

        if new_dist is not None and prev_dist is not None and new_dist < prev_dist:
            reward += 1.0

        return reward

    def distance_to_goal(self, x, y):
        dx = abs(int(x) - self.pos_farol[0])
        dy = abs(int(y) - self.pos_farol[1])
        dist = dx + dy
        max_dist = float(self.largura + self.altura) or 1.0
        return dist / max_dist


    def atualizacao(self):
        self.tempo += 1

    def reset(self):
        self.posicoes = {}
        self.tempo = 0

    def terminou(self, agentes=None):
        return all(getattr(a, "found_goal", False) for a in agentes)





