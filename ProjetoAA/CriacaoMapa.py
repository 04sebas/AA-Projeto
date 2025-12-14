import json

mapa = {
    "simulador": {
        "max_passos": 1000,
        "visualizacao": True
    },
    "ambiente": {
        "tipo": "AmbienteForaging",
        "largura": 50,
        "altura": 50,
        "recursos": [
            {"pos": [5, 5], "quantidade": 5, "valor": 10},
            {"pos": [40, 5], "quantidade": 8, "valor": 20},
            {"pos": [10, 20], "quantidade": 12, "valor": 8},
            {"pos": [25, 25], "quantidade": 10, "valor": 15},
            {"pos": [35, 35], "quantidade": 15, "valor": 25},
            {"pos": [5, 40], "quantidade": 10, "valor": 10},
            {"pos": [45, 45], "quantidade": 5, "valor": 30},

            {"pos": [20, 10], "quantidade": 7, "valor": 12},
            {"pos": [30, 15], "quantidade": 9, "valor": 18},
            {"pos": [15, 30], "quantidade": 14, "valor": 9},
            {"pos": [42, 28], "quantidade": 11, "valor": 14},
            {"pos": [10, 45], "quantidade": 6, "valor": 22},
            {"pos": [25, 5], "quantidade": 5, "valor": 11},
            {"pos": [45, 10], "quantidade": 10, "valor": 17}
        ],
        "ninhos": [
            [3, 3],
            [47, 47],
            [3, 47],
            [47, 3],
            [25, 25]
        ],
        "obstaculos": []
    },
    "agentes": [
        {
            "tipo": "AgenteFixo",
            "quantidade": 3,
            "posicao_inicial": "random",
            "politica": {
                "tipo": "greedy",
                "alcance": 5,
                "stuck_threshold": 3
            }
        },
        {
            "tipo": "AgenteAprendizagem",
            "quantidade": 1,
            "posicao_inicial": "random",
            "tipo_estrategia": "genetica",
            "sensores": 3,
            "trainable": True,
            "nome_base": "Genetico"
        }
    ]
}

with open("simulacao_foraging.json", "w") as f:
    json.dump(mapa, f, indent=4)
