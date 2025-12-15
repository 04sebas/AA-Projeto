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
                    quantidade = info.get("quantidade", 0)
                    if quantidade > 0:
                        visiveis.append({
                            "tipo": "recurso",
                            "pos": pos,
                            "valor": info.get("valor", 1),
                            "quantidade": quantidade
                        })
                    continue

                if hasattr(ambiente, "ninhos") and pos in ambiente.ninhos:
                    visiveis.append({"tipo": "ninho", "pos": pos})

                if hasattr(ambiente, "posicoes"):
                    from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
                    for agente, pos_ag in ambiente.posicoes.items():
                        if (
                            pos_ag == pos
                            and pos_ag != posicao_agente
                            and isinstance(agente, AgenteAprendizagem)
                        ):
                            visiveis.append({
                                "tipo": "agente",
                                "pos": pos,
                                "ref": agente
                            })
                            break
                    continue

        return visiveis
