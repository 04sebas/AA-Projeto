import random
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from Recolecao.AmbienteRecolecao import AmbienteRecolecao
import NeuralNetwork
from Recolecao.AgentRecolecao import AgenteRecolecao


class GeneticSimulation:

    def crossover(self, parent1, parent2, amb):

        """Performs one-point crossover between two parent agents."""
        point = random.randint(1, len(parent1.neural_network.weights) - 1)

        child1_geno = np.concatenate([parent1.neural_network.weights[:point], parent2.neural_network.weights[point:]])
        child2_geno = np.concatenate([parent2.neural_network.weights[:point], parent1.neural_network.weights[point:]])

        nn1 = NeuralNetwork.create_network_architecture(16, 4, (16,8 ))
        nn2 = NeuralNetwork.create_network_architecture(16, 4, (16,8 ))

        nn1.load_weights(child1_geno)
        nn2.load_weights(child2_geno)

        return AgenteRecolecao(amb, nn1), AgenteRecolecao(amb, nn2)

    def select_parent(self, population, tournament_size):

        """Selects a parent using tournament selection based on *combined_fitness*."""
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)

        return tournament[0]

    def geneticRun(self, plot_results):
        # --- EA Hyperparameters ---
        POPULATION_SIZE = 100
        NUM_GENERATIONS = 25
        MUTATION_RATE = 0.05
        TOURNAMENT_SIZE = 5

        # --- Initialization ---
        population = []
        last_generation = []
        amb = AmbienteRecolecao("Recolecao.txt")

        for _ in range(POPULATION_SIZE):
            nn = NeuralNetwork.create_network_architecture(16, 4, (16, 8))
            num_weights = nn.compute_num_weights()
            weights = [random.uniform(-1, 1) for _ in range(num_weights)]
            nn.load_weights(weights)
            agent = AgenteRecolecao(amb, nn)
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
                agent.setAmbiente(AmbienteRecolecao("Recolecao.txt"))

            for agent in population:
                agent.run_genetic_simulation()
                total_fitness += agent.getFitness()

            # 2. Sort population by *fitness*

            population.sort(key=lambda x: x.fitness, reverse=True)

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
                nn_copy = NeuralNetwork.create_network_architecture(16, 4, (16,8))
                nn_copy.load_weights(elite_agent.neural_network.weights.copy())
                new_agent = AgenteRecolecao(amb, nn_copy)  # Use MAIN farol
                new_population.append(new_agent)

            while len(new_population) < POPULATION_SIZE:
                parent1 = self.select_parent(population, TOURNAMENT_SIZE)  # This now uses combined_fitness
                parent2 = self.select_parent(population, TOURNAMENT_SIZE)

                child1, child2 = self.crossover(parent1, parent2, amb)

                child1.mutate(MUTATION_RATE) if random.random() < 0.7 else None
                child2.mutate(MUTATION_RATE) if random.random() < 0.7 else None

                new_population.append(child1)
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(child2)

            last_generation = population.copy()
            population = new_population

        np.savetxt("best_weights.txt", population[0].neural_network.weights)
        print("Evolution complete.")

        if plot_results:

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
            plot_gens = [0,5,9, 14, 19, 24]
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
            colors = cmap(np.linspace(0, 1, len(population)))

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