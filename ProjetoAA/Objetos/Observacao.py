class Observacao:
    def __init__(self, percepcoes=None):
        self.percepcoes = percepcoes
        self.posicao = (0, 0)
        self.largura = 0
        self.altura = 0

    def __repr__(self):
        if not self.percepcoes:
            return "Nada visível"
        return f"Observacao({self.percepcoes})"

    def contem(self, tipo_objeto: str) -> bool:
        return any(p["tipo"] == tipo_objeto for p in self.percepcoes)
