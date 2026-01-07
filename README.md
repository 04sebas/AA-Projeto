# Projeto AA

Um projeto baseado em agentedf autónomos com capacidades de aprendizagem (DQN, Genético).
Trabalhando em dois tipos de ambientes (Farol,Foraging).

## Estrutura

- `ProjetoAA/`: Código Source.
  - `Agents/`: Implementação dos Agentes (Fixos, Aprendizagem).
  - `Environments/`: Lógica dos Ambientes.
  - `Learning/`: Estratégias de Aprendizagem (DQN, Genético, Rede Neuronal).
  - `Objects/`: Objetos da Simulação (Sensores, Ações, Observações).
  - `SimulationEngine.py`: MotorDeSimulação.
  - `MapCreation.py`: Criação de Mapas.

## MotorDeSimulação

 Para executar a simulação, esta é feita no:
 ```bash
 SimulationEngine.py
 ```

 Vamos analisar o main:
 - Todos os passos com (Opcional), podem ser comentados caso não sejam utilizados.
 ```bash
 if __name__ == "__main__":
  simulator = SimulationEngine().create("simulador_foraging.json")
  if simulator.active:
      file_map = {
          2: "models/ForagingEnvironment_agent2_genetic_v1.pkl",
          3: "models/ForagingEnvironment_agent3_dqn_v1.pkl"
      }
      summary = simulator.load_networks_summary(file_map=file_map, agents=[2,3])
      results = simulator.run_experiments(num_runs=30, max_steps=750, file_map=file_map, seed=20, save_plot="results/aggregate.png")
      simulator.training_phase()
      simulator.testing_phase()
      simulator.save_animation_gif("models/trajectories_foraging.gif", fps=12, trail_len=30)
  ```

 i) Criação do Ambiente <Ficheiro.json>:
 ```bash
 simulator = SimulationEngine().create(<Ficheiro.json>)
```

 **(Opcional)**
 ii) Adicionar Agentes já treinados:  
 - Pode-se utilizar agentes já criados em .pk e assim adicionamos a este mapa, com o respetivo número [n1,n2,...].  
 - Caso adicione agentes com o seu index, também o terá que adicionar no agents=[n1,n2,...], nos argumentos do .load_networks_summary().  
 ```bash
 file_map = {
          2: "models/ForagingEnvironment_agent2_genetic_v1.pkl",
          3: "models/ForagingEnvironment_agent3_dqn_v1.pkl"
      }
 summary = simulator.load_networks_summary(file_map=file_map, agents=[2,3])
```
 
 **(Opcional)**
 iii) Múltiplas Simulações:  
 - Caso seja necessário testar várias simulações, com certos agentes definidos no passo ii) utiliza-se o run_experiments().  
 - Este apenas vai criar gráficos dos resultados finais, quantas vezes chegou ao Farol, ou quantos recursos foram Recolhidos e Depositados.  
```bash
 results = simulator.run_experiments(num_runs=30, max_steps=750, file_map=file_map, seed=20, save_plot="results/aggregate.png")
```
**(Opcional)**
 iv) Fase de treino para os agentes no <Ficheiro.json>:  
 ```bash
 simulator.training_phase()
 ```

 v) Fase de testes para os agentes anteriormente escolhidos (file_map, ou <Ficheiro.json>):  
 ```bash
 simulator.testing_phase()
```
 
**(Opcional)**
 vi) Criação de GIF:  
 - Criação de um gif para visualização do percurso dos agentes:  
 - Pode demorar um tempo se forem demasiados passos e altere o <nome_do_gif>.  
```bash
 simulator.save_animation_gif("models/<nome_do_gif>.gif", fps=12, trail_len=30)
```

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
