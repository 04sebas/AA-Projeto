from abc import ABC, abstractmethod
from ProjetoAA.Objetos.Sensor import Sensor
from ProjetoAA.Objetos.Accao import Accao

class Agente(ABC):
    def __init__(self, pos, nome, politica=None):
        self.pos = pos
        self.nome = nome
        self.politica = politica or {}
        
        self.recompensa_total = 0.0
        self.recursos_recolhidos = 0
        self.recursos_depositados = 0
        self.encontrou_objetivo = False
        self.treinavel = False
        
        self.ultima_obs = None
        
        alcance = self.politica.get("alcance", 1)
        self.sensores = Sensor(alcance=alcance)

    @classmethod
    @abstractmethod
    def cria(cls, nome_do_ficheiro_parametros: str):
        pass

    @abstractmethod
    def observacao(self, obs):
        self.ultima_obs = obs

    @abstractmethod
    def age(self) -> Accao:
        pass

    def avaliacao_estado_atual(self, recompensa: float):
        self.recompensa_total += recompensa

    def instala(self, sensor: Sensor):
        self.sensores.append(sensor)

    @abstractmethod
    def comunica(self, mensagem: str, de_agente):
        pass
