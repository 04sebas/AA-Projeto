from abc import ABC, abstractmethod
import random
from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Objetos.Observacao import Observacao


class Ambiente(ABC):
    def __init__(self, largura, altura, recursos=None, obstaculos=None, nome="Ambiente"):
        self.largura = int(largura)
        self.altura = int(altura)
        self.nome = nome
        self.tempo = 0
        self.posicoes = {}
        
        self.recursos = recursos or {}
        
        self.obstaculos = set()
        if obstaculos:
            self._interpretar_obstaculos(obstaculos)

    def _interpretar_obstaculos(self, obstaculos):
        for o in obstaculos:
            if isinstance(o, dict) and "pos" in o:
                self.obstaculos.add(tuple(o["pos"]))
            elif isinstance(o, (list, tuple)) and len(o) >= 2:
                self.obstaculos.add((int(o[0]), int(o[1])))

    @abstractmethod
    def observacao_para(self, agente: Agente) -> Observacao:
        pass

    @abstractmethod
    def agir(self, accao: Accao, agente: Agente):
        pass

    def atualizacao(self):
        self.tempo += 1

    def reiniciar(self):
        self.posicoes = {}
        self.tempo = 0

    @abstractmethod
    def terminou(self, agentes=None):
        pass

    def posicao_valida(self, x, y):
        if x < 0 or y < 0 or x >= self.largura or y >= self.altura:
            return False
        if (x, y) in self.obstaculos:
            return False
        return True

    def posicao_aleatoria(self):
        passo_x = self.largura // 4
        passo_y = self.altura // 4
        posicoes = []

        for r in range(1, 4):
            for c in range(1, 4):
                pos = (passo_x * c, passo_y * r)
                if pos in self.obstaculos or pos in self.recursos:
                    continue
                posicoes.append(pos)

        if not posicoes:
            tentativa = 0
            while True:
                pos = (random.randint(0, self.largura - 1), random.randint(0, self.altura - 1))
                if pos not in self.obstaculos and pos not in self.recursos:
                    return pos
                tentativa += 1
                if tentativa > 1000:
                    raise RuntimeError("Sem posições livres.")
        return random.choice(posicoes)
