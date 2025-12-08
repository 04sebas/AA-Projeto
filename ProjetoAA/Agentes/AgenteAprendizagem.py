from Agentes.Agente import Agente
from Objetos.Accao import Accao
from Objetos.Observacao import Observacao
from Aprendizagem.EstrategiaQLearning import EstrategiaQLearning

class AgenteAprendizagem(Agente):
    def __init__(self, tipo_estrategia="qlearning", neural_network=None, **kwargs):
        posicao_inicial = kwargs.get("posicao_inicial", [0, 0])
        super().__init__(posicao_inicial, "AA")
        self.neural_network = neural_network
        self.tipo_estrategia = tipo_estrategia

        if tipo_estrategia == "qlearning":
            self.estrategia = EstrategiaQLearning(
                alfa=kwargs.get("alfa", 0.1),
                gamma=kwargs.get("gamma", 0.9),
                epsilon=kwargs.get("epsilon", 0.1),
            )
        else:
            raise ValueError(f"Estratégia desconhecida: {tipo_estrategia}")

        self.estado_atual = None
        self.ultima_acao = None
        self.recompensa_total = 0

    def observacao(self, obs: Observacao):
        self.estado_atual = obs

    def cria(self, nome_do_ficheiro_parametros: str) -> "Agente":
        return self

    def age(self) -> Accao:
        acoes_possiveis = ["cima", "baixo", "esquerda", "direita"]
        acao = self.estrategia.escolher_acao(self.estado_atual, acoes_possiveis)
        self.ultima_acao = acao

        if self.estado_atual is not None and hasattr(self.estado_atual, "posicao"):
            self.pos = list(self.estado_atual.posicao)

        return Accao(acao)

    def avaliacao_estado_atual(self, recompensa: float):
        self.estrategia.atualizar(self.estado_atual, self.ultima_acao, recompensa, self.estado_atual)
        self.recompensa_total += recompensa

    def comunica(self, mensagem: str, de_agente: "Agente"):
        print(f"[{self.nome}] recebeu de {de_agente.nome}: {mensagem}")
