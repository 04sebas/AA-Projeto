import json

map_config = {
    "simulator": {
        "max_steps": 1000,
        "visualization": True
    },
    "environment": {
        "type": "ForagingEnvironment",
        "width": 50,
        "height": 50,
        "resources": [
            {"pos": [5, 5], "quantity": 5, "value": 10},
            {"pos": [40, 5], "quantity": 8, "value": 20},
            {"pos": [10, 20], "quantity": 12, "value": 8},
            {"pos": [25, 25], "quantity": 10, "value": 15},
            {"pos": [35, 35], "quantity": 15, "value": 25},
            {"pos": [5, 40], "quantity": 10, "value": 10},
            {"pos": [45, 45], "quantity": 5, "value": 30},

            {"pos": [20, 10], "quantity": 7, "value": 12},
            {"pos": [30, 15], "quantity": 9, "value": 18},
            {"pos": [15, 30], "quantity": 14, "value": 9},
            {"pos": [42, 28], "quantity": 11, "value": 14},
            {"pos": [10, 45], "quantity": 6, "value": 22},
            {"pos": [25, 5], "quantity": 5, "value": 11},
            {"pos": [45, 10], "quantity": 10, "value": 17}
        ],
        "nests": [
            [3, 3],
            [47, 47],
            [3, 47],
            [47, 3],
            [25, 25]
        ],
        "obstacles": []
    },
    "agents": [
        {
            "type": "FixedAgent",
            "quantity": 3,
            "initial_position": "random",
            "policy": {
                "type": "greedy",
                "range": 5,
                "stuck_threshold": 3
            }
        },
        {
            "type": "LearningAgent",
            "quantity": 1,
            "initial_position": "random",
            "strategy_type": "genetic",
            "sensors": 3,
            "trainable": True,
            "base_name": "Genetic"
        }
    ]
}

with open("simulator_foraging.json", "w") as f:
    json.dump(map_config, f, indent=4)
