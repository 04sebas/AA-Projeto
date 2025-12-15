import random
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import argmax

from Projeto.Agentes.AgenteAprendizagem import AgenteAprendizagem
from Projeto.Ambientes.AmbienteRecolecao import AmbienteRecolecao
from Projeto.Aprendizagem import NeuralNetwork
from Projeto.Ambientes.AmbienteFarol import AmbienteFarol
from Projeto.Objetos.ObjetosAmbiente import Goal


class EstrategiaGenetica:
    def __init__(self):
        pass

    @staticmethod
    def crossover( parent1, parent2, amb, inputSize):

        """Performs one-point crossover between two parent agents."""
        point = random.randint(1, len(parent1.neural_network.weights) - 1)

        child1_geno = np.concatenate([parent1.neural_network.weights[:point], parent2.neural_network.weights[point:]])
        child2_geno = np.concatenate([parent2.neural_network.weights[:point], parent1.neural_network.weights[point:]])

        nn1 = NeuralNetwork.create_network_architecture(inputSize, 4, (16, 8))
        nn2 = NeuralNetwork.create_network_architecture(inputSize, 4, (16, 8))

        nn1.load_weights(child1_geno)
        nn2.load_weights(child2_geno)

        return AgenteAprendizagem(amb, nn1), AgenteAprendizagem(amb, nn2)

    @staticmethod
    def select_parent( population, tournament_size):

        """Selects a parent using tournament selection based on *combined_fitness*."""
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)

        return tournament[0]

    def algoritmoGenetico(self, fileName, inputSize, typeAmb):
        # --- EA Hyperparameters ---
        POPULATION_SIZE = 100
        NUM_GENERATIONS = 50
        MUTATION_RATE = 0.05
        TOURNAMENT_SIZE = 5

        # --- Initialization ---
        population = []
        last_generation = []

        amb = None

        for _ in range(POPULATION_SIZE):
            if typeAmb == "Farol":
                amb = AmbienteFarol(fileName)
            elif typeAmb == "Recolecao":
                amb = AmbienteRecolecao(fileName)
            else:
                print("Type incorrect.")
                return
            nn = NeuralNetwork.create_network_architecture(inputSize, 4, (16, 8))
            num_weights = nn.compute_num_weights()
            weights = [random.uniform(-1, 1) for _ in range(num_weights)]
            nn.load_weights(weights)
            agent = AgenteAprendizagem(amb, nn)
            population.append(agent)
        avg_fitness_per_gen = []
        best_paths_per_gen = []

        print("Starting evolution...")

        # --- Generational Loop ---
        for gen in range(NUM_GENERATIONS):

            print(f"Generation {gen + 1}/{NUM_GENERATIONS}")
            total_fitness = 0

            # 1. Evaluate Population

            for agent in population:
                agent.reset()
                start_x, start_y = agent.amb.random_valid_position()
                agent.setPosition(start_x, start_y)
                ##ALGORITMO
                if amb.type == "Farol":
                    agent.currentGoal = Goal(agent.amb.goalx, agent.amb.goaly)

                agent.behavior.add((agent.x, agent.y))
                agent.path.append((agent.x, agent.y))

                for _ in range(agent.num_steps):

                    obs = agent.observacao(3)
                    agent.update_goal_from_observation()

                    if agent.currentGoal is None:
                        if not agent.carrying:
                            if agent.resources:
                                goals = list(agent.resources)
                                agent.currentGoal = agent.choose_closest(goals)

                            else:
                                agent.randomStepNum += 1

                                stepReward, _, done= agent.randomStep()
                                agent.behavior.add((agent.x, agent.y))
                                agent.path.append((agent.x, agent.y))

                                continue
                        else:
                            if agent.delivery:
                                goals = list(agent.delivery)
                                agent.currentGoal = agent.choose_closest(goals)
                            else:
                                agent.randomStepNum += 1
                                stepReward, _ = agent.randomStep()
                                agent.behavior.add((agent.x, agent.y))
                                agent.path.append((agent.x, agent.y))
                                continue

                    current_distance = abs(agent.x - agent.currentGoal.x) + abs(agent.y - agent.currentGoal.y)

                    if agent.amb.type == "Recolecao":
                        if agent.amb.lastGoalDistance is None or current_distance < agent.amb.lastGoalDistance:
                            agent.amb.lastGoalDistance = current_distance
                            agent.amb.goalStepsWithoutProgress = 0
                        else:
                            agent.amb.goalStepsWithoutProgress += 1

                        if agent.amb.goalStepsWithoutProgress >= agent.amb.maxGoalSteps:
                            agent.amb.currentGoal = None
                            agent.amb.goalStepsWithoutProgress = 0
                            agent.amb.lastGoalDistance = None
                            continue

                    goalx, goaly = agent.currentGoal.x / agent.amb.size, agent.currentGoal.y / agent.amb.size

                    if amb.type == "Farol":
                        inputs = np.array([agent.x, agent.y, *agent.obs, agent.distanceToGoal(agent.x, agent.y)])
                    else:
                        inputs = np.array([agent.x / agent.amb.size,agent.y / agent.amb.size,*obs, goalx, goaly, ])

                    output_vector = agent.neural_network.forward(inputs)
                    action_index = argmax(output_vector)
                    stepReward, done = agent.age(action_index)
                    agent.behavior.add((agent.x, agent.y))
                    agent.path.append((agent.x, agent.y))

                    if done:
                        agent.founGoal = True
                        break

                total_fitness += agent.getFitness()

            # 2. Sort population by *fitness*

            population.sort(key=lambda x: x.fitness, reverse=True)

            if typeAmb == "Farol":
                if population[0].foundGoal:
                    print(f" Best agent found the goal!")
                i = 0
                for _ in range(POPULATION_SIZE):
                    if population[_].foundGoal:
                        i += 1
                print(f" Agents that found the goal: {i}")
            else:
                print(f"Best agent found: {population[0].delivered}")

            # 3. Log results for this generation
            avg_fitness = total_fitness / POPULATION_SIZE
            best_paths_per_gen.append(population[0].path)
            print(
                f"Gen {gen + 1}/{NUM_GENERATIONS} | Avg Fitness: {avg_fitness:.2f} | Best Fitness: {population[0].fitness:.2f})")

            avg_fitness_per_gen.append(avg_fitness)

            # 4. Create new generation (Selection, Crossover, Mutation)
            new_population = []

            n_elite = POPULATION_SIZE // 5

            for elite_agent in population[:n_elite]:
                nn_copy = NeuralNetwork.create_network_architecture(inputSize, 4, (16, 8))
                nn_copy.load_weights(elite_agent.neural_network.weights.copy())
                new_agent = AgenteAprendizagem(amb, nn_copy)  # Use MAIN farol
                new_population.append(new_agent)

            while len(new_population) < POPULATION_SIZE:
                t = TOURNAMENT_SIZE
                parent1 = self.select_parent(population, t)  # This now uses combined_fitness
                parent2 = self.select_parent(population, t)

                child1, child2 = self.crossover(parent1, parent2, amb, inputSize)

                child1.mutate(MUTATION_RATE) if random.random() < 0.7 else None
                child2.mutate(MUTATION_RATE) if random.random() < 0.7 else None

                new_population.append(child1)
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(child2)

            last_generation = population.copy()
            population = new_population
        print("Evolution complete.")
        if typeAmb == "Farol":
            np.savetxt("../Projeto/MelhoresModelos/melhorFarolGenetico.txt", population[0].neural_network.weights)
            self.plotFarol(best_paths_per_gen, amb, NUM_GENERATIONS, avg_fitness_per_gen, last_generation)
        else:
            np.savetxt("../Projeto/MelhoresModelos/melhorRecolecaoGenetico.txt", population[0].neural_network.weights)
            self.plotRecolecao(best_paths_per_gen, amb, NUM_GENERATIONS, avg_fitness_per_gen, last_generation)

    def plotFarol(self, best_paths_per_gen, amb, NUM_GENERATIONS, avg_fitness_per_gen, last_generation):
        ##### PATH PLOT #####
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = cm.rainbow(np.linspace(0, 1, len(best_paths_per_gen)))

        # Plot walls and goal
        for wall in amb.walls:
            ax.add_patch(
                patches.Rectangle(
                    (wall.x - 0.5, wall.y - 0.5),  # lower-left corner
                    1, 1,  # width, height
                    facecolor='black'
                )
            )

        ax.text(amb.goalx, amb.goaly, "G", color='green', fontsize=9, ha='center', va='center',
                fontweight='bold')
        # Plot paths
        plot_gens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     NUM_GENERATIONS - 1]
        for i in plot_gens:
            path = best_paths_per_gen[i]
            avg_fitness = avg_fitness_per_gen[i]  # Get the avg combined fitness
            x_vals = [p[0] for p in path]
            y_vals = [p[1] for p in path]
            ax.plot(x_vals, y_vals, color=colors[i], label=f"Gen {i + 1} (Avg Fitness: {avg_fitness:.2f})",
                    alpha=0.7)
            ax.plot(x_vals[-1], y_vals[-1], 'x', color=colors[i], markersize=10, markeredgewidth=2)

        ax.set_xlim(-1, 100)
        ax.set_ylim(-1, 100)
        ax.set_title("Best Agent Paths Over Generations")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)

        ##### Fiteness Plot #####

        plt.figure(figsize=(10, 5))
        plt.plot(range(NUM_GENERATIONS), avg_fitness_per_gen, marker='o')
        plt.title("Average Combined Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Average Combined Fitness Score")
        plt.grid(True)

        ##### PLOT ALL AGENTS PATHS FROM LAST GENERATION #####
        print("Plotting all agents from last generation...")

        fig2, ax2 = plt.subplots(figsize=(10, 10))

        # Plot walls
        for wall in amb.walls:
            ax2.add_patch(
                patches.Rectangle(
                    (wall.x - 0.5, wall.y - 0.5),
                    1, 1,
                    facecolor='black'
                )
            )

        # Goal
        ax2.text(
            amb.goalx, amb.goaly, "G",
            color='green', fontsize=10,
            ha='center', va='center', fontweight='bold'
        )

        # --- Use get_cmap() ---
        cmap = plt.get_cmap("viridis")  # You can choose ANY cmap here
        colors = cmap(np.linspace(0, 1, len(last_generation)))

        # Plot each agent path
        for i, agent in enumerate(last_generation):
            if hasattr(agent, "path") and len(agent.path) > 1:
                x_vals = [p[0] for p in agent.path]
                y_vals = [p[1] for p in agent.path]

                ax2.plot(
                    x_vals, y_vals,
                    color=colors[i],
                    alpha=0.5,
                    linewidth=1
                )

                # Mark final position
                ax2.plot(
                    x_vals[-1], y_vals[-1],
                    marker='x',
                    color=colors[i],
                    markersize=6
                )

        ax2.set_xlim(-1, 100)
        ax2.set_ylim(-1, 100)
        ax2.set_title("All Agent Paths - Last Generation")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True)
        plt.show()


    def plotRecolecao(self, best_paths_per_gen, amb, NUM_GENERATIONS, avg_fitness_per_gen, last_generation):
        ##### PATH PLOT #####
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = cm.rainbow(np.linspace(0, 1, len(best_paths_per_gen)))

        # Plot walls and goal
        for wall in amb.walls:
            ax.add_patch(
                patches.Rectangle(
                    (wall.x - 0.5, wall.y - 0.5),  # lower-left corner
                    1, 1,  # width, height
                    facecolor='black'
                )
            )

        for resource in amb.resources:
            ax.text(resource.x, resource.y, "R", color='green', fontsize=9, ha='center', va='center',
                    fontweight='bold')

        for delivery in amb.deliveryPoints:
            ax.text(delivery.x, delivery.y, "P", color='red', fontsize=9, ha='center', va='center',
                    fontweight='bold')

        for agente in amb.agentes:
            ax.text(agente.x, agente.y, "A", color='blue', fontsize=9, ha='center', va='center', fontweight='bold')

        # Plot paths
        plot_gens = [0, 10, 20, 30, 40, 49]
        for i in plot_gens:
            path = best_paths_per_gen[i]
            avg_fitness = avg_fitness_per_gen[i]  # Get the avg combined fitness
            x_vals = [p[0] for p in path]
            y_vals = [p[1] for p in path]
            ax.plot(x_vals, y_vals, color=colors[i], label=f"Gen {i + 1} (Avg Fitness: {avg_fitness:.2f})",
                    alpha=0.7)
            ax.plot(x_vals[-1], y_vals[-1], 'x', color=colors[i], markersize=10, markeredgewidth=2)

        ax.set_xlim(-1, 50)
        ax.set_ylim(-1, 50)
        ax.set_title("Best Agent Paths Over Generations")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)

        ##### Fiteness Plot #####

        plt.figure(figsize=(10, 5))
        plt.plot(range(NUM_GENERATIONS), avg_fitness_per_gen, marker='o')
        plt.title("Average Combined Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Average Combined Fitness Score")
        plt.grid(True)

        ##### PLOT ALL AGENTS PATHS FROM LAST GENERATION #####
        print("Plotting all agents from last generation...")

        fig2, ax2 = plt.subplots(figsize=(10, 10))

        # Plot walls
        for wall in amb.walls:
            ax2.add_patch(
                patches.Rectangle(
                    (wall.x - 0.5, wall.y - 0.5),
                    1, 1,
                    facecolor='black'
                )
            )

        for resource in amb.resources:
            ax2.text(resource.x, resource.y, "R", color='green', fontsize=9, ha='center', va='center',
                     fontweight='bold')

        for delivery in amb.deliveryPoints:
            ax2.text(delivery.x, delivery.y, "P", color='red', fontsize=9, ha='center', va='center',
                     fontweight='bold')

        for agente in amb.agentes:
            ax2.text(agente.x, agente.y, "A", color='blue', fontsize=9, ha='center', va='center', fontweight='bold')

        # --- Use get_cmap() ---
        cmap = plt.get_cmap("viridis")  # You can choose ANY cmap here
        colors = cmap(np.linspace(0, 1, len(last_generation)))

        # Plot each agent path
        for i, agent in enumerate(last_generation):
            if hasattr(agent, "path") and len(agent.path) > 1:
                x_vals = [p[0] for p in agent.path]
                y_vals = [p[1] for p in agent.path]

                ax2.plot(
                    x_vals, y_vals,
                    color=colors[i],
                    alpha=0.5,
                    linewidth=1
                )

                # Mark final position
                ax2.plot(
                    x_vals[-1], y_vals[-1],
                    marker='x',
                    color=colors[i],
                    markersize=6
                )

        ax2.set_xlim(-1, 50)
        ax2.set_ylim(-1, 50)
        ax2.set_title("All Agent Paths - Last Generation")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True)
        plt.show()