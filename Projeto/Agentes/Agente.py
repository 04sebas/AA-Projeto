from abc import ABC, abstractmethod

class Agente(ABC):
    def __init__(self, pos, amb):
        self.recompensa_total = 0.0
        self.pos = pos
        self.randomStepNum = 0
        self.y = None
        self.x = None
        self.yInitial = None
        self.xInitial = None
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.num_steps = 1500
        self.amb = amb
        self.behavior = set()
        self.path = []
        self.resources = set()
        self.delivery = set()
        self.carrying = False
        self.currentGoal = None
        self.delivered = 0
        self.done = False
        self.obs = None


    @classmethod
    @abstractmethod
    def cria(cls, nome_do_ficheiro_parametros):
        pass

    @abstractmethod
    def observacao(self,  depth):
        pass

    @abstractmethod
    def age(self):
        pass

    @abstractmethod
    def avaliacao_estado_atual(self, recompensa):
        pass

    @abstractmethod
    def comunica(self, mensagem, de_agente):
        #Método genérico de comunicação entre agentes (opcional)
        pass
