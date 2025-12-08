class Sensor:
    def __init__(self, alcance: int = 3):
        self.alcance = alcance

    def perceber(self, ambiente, posicao_agente):
        visiveis = []
        x, y = posicao_agente

        for dx in range(-self.alcance, self.alcance + 1):
            for dy in range(-self.alcance, self.alcance + 1):
                pos = (x + dx, y + dy)

                if not ambiente.posicao_valida(*pos):
                    continue

                if pos == (x, y):
                    continue

                if pos in ambiente.obstaculos:
                    visiveis.append({"tipo": "obstaculo", "pos": pos})
                    continue

                if pos in ambiente.recursos:
                    info = ambiente.recursos[pos]
                    visiveis.append({
                        "tipo": "recurso",
                        "pos": pos,
                        "valor": info.get("valor", 1),
                        "qtd": info.get("quantidade", 1)
                    })
                    continue

                if hasattr(ambiente, "ninhos") and pos in ambiente.ninhos:
                    visiveis.append({"tipo": "ninho", "pos": pos})

        return visiveis
