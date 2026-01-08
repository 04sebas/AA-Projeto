# Projeto AA

Um projeto baseado em agentes autónomos com capacidades de aprendizagem (DQN, Genético).  
Trabalhando em dois tipos de ambientes (Farol,Foraging).

## Estrutura

- `Agents/`: Implementação dos Agentes (Fixos, Aprendizagem).
- `Environments/`: Lógica dos Ambientes.
- `Learning/`: Estratégias de Aprendizagem (DQN, Genético, Rede Neuronal, Políticas).
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

## Configuração (`.json`):

O MotorDeSimulação utiliza ficheiros JSON para definir os Ambientes e Agentes:
*   **`environment`**: Tipo (`FarolEnvironment`, `ForagingEnvironment`), dimensões, e objetos.
*   **`agents`**: Lista de agentes, os seus tipos (`LearningAgent`, `FixedAgent`), quantidade e definições da estratégia.
*   **`simulator`**: Definições globais do tipo `max_steps` e `visualization`.

**Exemplo de Ambiente:**
```bash
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
```
