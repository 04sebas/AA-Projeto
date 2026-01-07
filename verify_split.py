
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from ProjetoAA.SimulationEngine import SimulationEngine

# Create a config for testing
config_content = """
{
  "environment": {
    "type": "LighthouseEnvironment",
    "width": 20,
    "height": 20,
    "lighthouse_pos": [15, 15],
    "obstacles": []
  },
  "agents": [
    {
      "type": "LearningAgent",
      "quantity": 1,
      "trainable": true,
      "strategy_type": "genetic",
      "strategy_conf": {
        "num_generations": 2,
        "population_size": 4,
        "trials_per_evaluation": 2
      }
    }
  ],
  "simulator": {
    "max_steps": 50,
    "visualization": false
  }
}
"""

with open("test_verify_config.json", "w", encoding="utf-8") as f:
    f.write(config_content)

print("[VERIFY] Creating simulation...")
sim = SimulationEngine().create("test_verify_config.json")

print("[VERIFY] Running training_phase...")
# This should generate test_dataset.json and split data
sim.training_phase()

if os.path.exists("test_dataset.json"):
    print("[VERIFY] SUCCESS: test_dataset.json created.")
else:
    print("[VERIFY] FAILURE: test_dataset.json NOT created.")

print("[VERIFY] Running testing_phase...")
# This should load test_dataset.json
sim.testing_phase()

print("[VERIFY] Done.")
