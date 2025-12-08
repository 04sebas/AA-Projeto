from abc import ABC, abstractmethod

class EstrategiaAprendizagem(ABC):
    @abstractmethod
    def escolher_acao(self, estado, acoes_possiveis):
        pass
