# AA Project

This project is an agent-based simulation framework with learning capabilities (DQN, Genetic Algorithms).
It works with different environments (Lighthouse, Foraging).

## Structure

- `aa_project/`: Source code.
  - `agents/`: Agent implementations (Fixed, Learning).
  - `environments/`: Environment logic.
  - `learning/`: Learning strategies (DQN, Genetic, Neural Network).
  - `objects/`: Simulation objects (Sensor, Action, Observation).
  - `simulation_engine.py`: Main engine.
  - `map_creation.py`: Script to generate configuration files.

## Running the Simulation

1. **Generate Configuration**:
   Run `aa_project/map_creation.py` to generate `simulator_foraging.json`.
   ```bash
   python aa_project/map_creation.py
   ```

2. **Run Simulation**:
   You can run the simulation engine directly or via a script.
   To verify the setup and training split:
   ```bash
   python verify_split.py
   ```

## SimulationEngine

The `SimulationEngine` is the core class responsible for managing the simulation lifecycle. It orchestrates the environment, agents, and learning processes.

### Key Methods:

*   **`create(param_filename)`**: Initializes the simulation using a JSON configuration file. It sets up the environment (width, height, obstacles), agents (types, policies), and strategy configurations.
*   **`training_phase()`**: Executes the training loop for learning agents.
    *   Generates a 70/30 train/test split of initial positions.
    *   Trains agents using the specified strategy (e.g., Genetic Algorithm, DQN) using *only* the training set.
    *   Saves the best performing neural network models to the `models/` directory.
    *   Automatically validates performance on the test set to ensure generalization.
*   **`testing_phase()`**: Loads the test dataset (unseen during training) and evaluates the agents' performance.
    *   Crucial for verifying that agents haven't just memorized the training positions.
    *   Produces performance plots and fitness statistics.
*   **`run(max_steps)`**: Runs a single simulation episode (usually for visualization or debugging).
*   **`visualize_paths()`**: Generates plots showing agent trajectories, resources collected, and environment layout using Matplotlib.

### Configuration (`.json`):

The engine depends on a JSON config file defining:
*   **`environment`**: Type (`LighthouseEnvironment`, `ForagingEnvironment`), dimensions, and objects.
*   **`agents`**: List of agents, their types (`LearningAgent`, `FixedAgent`), quantity, and strategy settings.
*   **`simulator`**: Global settings like `max_steps` and `visualization`.

## Features

- **Environments**: Lighthouse (Navigation), Foraging (Resource Collection).
- **Learning**:
  - **DQN**: Deep Q-Learning with Double DQN and Experience Replay.
  - **Genetic**: Genetic Algorithm with Tournament Selection, Elitism, and multiple evaluation trials.
- **Split**: Automatic 70/30 Train/Test split for generalization testing.

## Recent Updates

- Translated codebase from Portuguese to English.
- Implemented Double DQN for stability.
- Improved Genetic Algorithm with averaged evaluations.
- Added automatic Test Set generation.
