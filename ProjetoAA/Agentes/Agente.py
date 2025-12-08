from abc import ABC, abstractmethod
from Objetos.Accao import Accao
from Objetos.Observacao import Observacao
from Objetos.Sensor import Sensor


class Agente(ABC):
    def __init__(self, pos,nome):
        self.sensores = Sensor()
        self.recompensa_total = 0.0
        self.pos = pos
        self.nome = nome

    @classmethod
    @abstractmethod
    def cria(cls, nome_do_ficheiro_parametros: str) -> "Agente":
        """
        Cria e configura um agente com base num ficheiro de parâmetros.
        """
        pass

    @abstractmethod
    def observacao(self, obs: Observacao):
        """
        Recebe a observação do ambiente e atualiza o estado interno do agente.
        """
        pass

    @abstractmethod
    def age(self) -> Accao:
        """
        Decide e devolve a ação a executar.
        """
        pass

    @abstractmethod
    def avaliacao_estado_atual(self, recompensa: float):
        """
        Atualiza o agente com a recompensa recebida do ambiente.
        """
        pass

    def instala(self, sensor: Sensor):
        """
        Instala um sensor (objeto que fornece percepções ao agente).
        """
        self.sensores.append(sensor)

    @abstractmethod
    def comunica(self, mensagem: str, de_agente: "Agente"):
        """
        Método genérico de comunicação entre agentes (opcional).
        """
        pass
