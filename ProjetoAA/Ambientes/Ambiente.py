from abc import ABC, abstractmethod

from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Objetos.Observacao import Observacao


class Ambiente(ABC):
    def __init__(self, largura, altura, recursos=None, obstaculos=None):
        self.largura = largura
        self.altura = altura
        self.recursos = recursos or []
        self.obstaculos = obstaculos or []
        self.posicoes = {}
        self.tempo = 0


    @abstractmethod
    def observacao_para(self, agente: Agente) -> Observacao:
        """
        Retorna a observação do ambiente para o agente indicado.
        A observação depende do tipo de ambiente (ex: visão parcial, posição, etc.).
        """
        pass

    @abstractmethod
    def agir(self, accao: Accao, agente: Agente):
        """
        Executa a ação no ambiente e devolve a recompensa obtida.
        Cada ambiente define o seu próprio modelo de transição e recompensa.
        """
        pass

    @abstractmethod
    def atualizacao(self):
        """
        Atualiza o estado global do ambiente (recursos, tempo, eventos, etc.).
        Chamado a cada passo de simulação.
        """
        pass

    def posicao_valida(self, x, y):
        """
        Verifica se a posição está dentro dos limites e não contém obstáculos.
        """
        if x < 0 or y < 0 or x >= self.largura or y >= self.altura:
            return False
        if (x, y) in self.obstaculos:
            return False
        return True


