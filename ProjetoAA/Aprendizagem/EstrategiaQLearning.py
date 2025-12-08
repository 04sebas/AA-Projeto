import random
from Aprendizagem.EstrategiaAprendizagem import EstrategiaAprendizagem

class EstrategiaQLearning(EstrategiaAprendizagem):
    def __init__(self, alfa=0.1, gamma=0.9, epsilon=0.1):
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def escolher_acao(self, estado, acoes_possiveis):
        if random.random() < self.epsilon:
            return random.choice(acoes_possiveis)
        q_values = [self.q_table.get((estado, a), 0.0) for a in acoes_possiveis]
        max_q = max(q_values)
        melhores = [a for a, q in zip(acoes_possiveis, q_values) if q == max_q]
        return random.choice(melhores)

    def atualizar(self, estado, acao, recompensa, novo_estado):
        chave = (estado, acao)
        valor_atual = self.q_table.get(chave, 0.0)
        acoes_possiveis = ["cima", "baixo", "esquerda", "direita"]
        max_q = max(self.q_table.get((novo_estado, a), 0.0) for a in acoes_possiveis)
        self.q_table[chave] = valor_atual + self.alfa * (
            recompensa + self.gamma * max_q - valor_atual
        )
