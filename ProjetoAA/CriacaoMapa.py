import json

mapa = {
    "simulador": {
        "max_passos": 1000,
        "visualizacao": True
    },
    "ambiente": {
        "tipo": "AmbienteForaging",
        "largura": 100,
        "altura": 100,
        "recursos": [
            {"pos": [10, 10], "quantidade": 5, "valor": 10},
            {"pos": [80, 10], "quantidade": 8, "valor": 20},
            {"pos": [20, 40], "quantidade": 15, "valor": 5},
            {"pos": [50, 50], "quantidade": 10, "valor": 15},
            {"pos": [75, 75], "quantidade": 20, "valor": 25},
            {"pos": [10, 80], "quantidade": 10, "valor": 10},
            {"pos": [90, 90], "quantidade": 5, "valor": 30}
        ],
        "ninhos": [
            [5, 5], [95, 95]
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
                "alcance": 15,
                "stuck_threshold": 3
            }
        },
        {
            "tipo": "AgenteFixo",
            "quantidade": 2,
            "posicao_inicial": "random",
            "politica": {
                "tipo": "random",
                "alcance": 5
            }
        }
    ]
}

obstaculos = []

obstaculos += [{"pos": [i, 25]} for i in range(50, 75)]
obstaculos += [{"pos": [i, 75]} for i in range(10, 50)]

obstaculos += [{"pos": [25, i]} for i in range(30, 75)]
obstaculos += [{"pos": [75, i]} for i in range(20, 80)]

mapa["ambiente"]["obstaculos"] = obstaculos

with open("simulacao_foraging.json", "w") as f:
    json.dump(mapa, f, indent=4)
