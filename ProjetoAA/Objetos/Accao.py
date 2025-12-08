class Accao:
    def __init__(self, nome, parametros=None):
        self.nome = nome or None
        self.parametros = parametros or {}

    def __repr__(self):
        return f"Accao({self.parametros})"
