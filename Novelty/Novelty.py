import random
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from Farol import Farol
from Farol.AgentFarol import AgenteLearner

class Novelty:
    def crossover(self, parent1, parent2, farol):
        """Performs one-point crossover between two parent agents."""
        point = random.randint(1, len(parent1.genotype) - 1)
        child1_geno = parent1.genotype[:point] + parent2.genotype[point:]
        child2_geno = parent2.genotype[:point] + parent1.genotype[point:]
        return AgenteLearner(farol, child1_geno), AgenteLearner(farol, child2_geno)

    def select_parent(self, population, tournament_size):
        """Selects a parent using tournament selection based on *combined_fitness*."""
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: x.combined_fitness, reverse=True)
        return tournament[0]
    
    def jaccard_distance(self, set1, set2):
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return 1 - intersection / union if union != 0 else 0

    def compute_novelty(self, current_behavior, archive, k=5):
        # Handle the empty archive case
        if not archive:
            # The first item is, by definition, maximally novel
            return 1.0

        distances = [self.jaccard_distance(current_behavior, b) for b in archive]
        distances.sort()

        # Your original logic is now safe because we know len(distances) > 0
        return sum(distances[:k]) / k if len(distances) >= k else sum(distances) / len(distances)

    def noveltyRun(self):
            # --- EA Hyperparameters ---
        POPULATION_SIZE = 100
        NUM_GENERATIONS = 25
        MUTATION_RATE = 0.01
        TOURNAMENT_SIZE = 3
        N_ARCHIVE_ADD = 5  # Add top 5 most novel agents to archive each gen

    # --- Initialization ---
        archive = []
        population = []
        farol = Farol("../Farol.txt")
        for _ in range(POPULATION_SIZE):
            agent = AgenteLearner(farol)
            population.append(agent)
        avg_fitness_per_gen = []
        best_paths_per_gen = []

        print("Starting evolution...")

# --- Generational Loop ---
        for gen in range(NUM_GENERATIONS):
            print(f"Generation {gen+1}/{NUM_GENERATIONS}")
            total_fitness = 0

            # 1. Evaluate Population
            for agent in population:
                agent.run_simulation()

                # --- Calculate and combine scores ---
                novelty_score = self.compute_novelty(agent.behavior, archive)
                objective_score = agent.calculate_objective_fitness()

                # Combine the scores.
                # You might need to add a weight, e.g.:
                novelty_weight = 1000 # Make novelty competitive with fitness
                agent.combined_fitness = (novelty_score * novelty_weight) + objective_score

                #agent.combined_fitness = novelty_score #+ objective_score

                total_fitness += agent.combined_fitness

            # 2. Sort population by *combined_fitness*
            population.sort(key=lambda x: x.combined_fitness, reverse=True)
            if population[0].foundGoal:
                print(f"  Best agent found the goal!")
            # 3. Log results for this generation
            avg_fitness = total_fitness / POPULATION_SIZE
            avg_fitness_per_gen.append(avg_fitness)
            best_paths_per_gen.append(population[0].path)

            # Get the top agent's individual scores for logging
            best_nov = self.compute_novelty(population[0].behavior, archive)
            best_obj = population[0].calculate_objective_fitness()

            print(f"Gen {gen+1}/{NUM_GENERATIONS} | Avg Combined: {avg_fitness:.2f} | Best Combined: {population[0].combined_fitness:.2f} (Nov: {best_nov:.2f}, Obj: {best_obj})")

            # 4. Update archive with the most novel behaviors (from this gen)
            #    We still update the archive based on *pure novelty*

            # Sort by novelty just for archive update
            population.sort(key=lambda x: self.compute_novelty(x.behavior, archive), reverse=True)
            for i in range(N_ARCHIVE_ADD):
                archive.append(population[i].behavior)

            # Re-sort by combined fitness for breeding
            population.sort(key=lambda x: x.combined_fitness, reverse=True)

            # 5. Create new generation (Selection, Crossover, Mutation)
            new_population = []

            n_elite = POPULATION_SIZE // 10
            new_population.extend(population[:n_elite])
    
            while len(new_population) < POPULATION_SIZE:
                parent1 = self.select_parent(population, TOURNAMENT_SIZE) # This now uses combined_fitness
                parent2 = self.select_parent(population, TOURNAMENT_SIZE)

                child1, child2 = self.crossover(parent1, parent2, farol)

                child1.mutate(MUTATION_RATE)
                child2.mutate(MUTATION_RATE)

                new_population.append(child1)
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(child2)

            population = new_population

        print("Evolution complete.")
    
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = cm.rainbow(np.linspace(0, 1, len(best_paths_per_gen)))
        
        # Plot walls and goal
        for wall in farol.walls:
            ax.add_patch(
            patches.Rectangle(
                (wall.x - 0.5, wall.y - 0.5),   # lower-left corner
                    1, 1,                          # width, height
                facecolor='black'
                )
            )
        # Plot paths
        plot_gens = [0, NUM_GENERATIONS // 2, NUM_GENERATIONS - 1]
        for i in plot_gens:
            path = best_paths_per_gen[i]
            avg_fitness = avg_fitness_per_gen[i] # Get the avg combined fitness
            x_vals = [p[0] for p in path]
            y_vals = [p[1] for p in path]
            ax.plot(x_vals, y_vals, color=colors[i], label=f"Gen {i+1} (Avg Fitness: {avg_fitness:.2f})", alpha=0.7)
            ax.plot(x_vals[-1], y_vals[-1], 'x', color=colors[i], markersize=10, markeredgewidth=2)

        ax.set_xlim(-1, 100)
        ax.set_ylim(-1, 100)
        ax.set_title("Best Agent Paths Over Generations")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)
        #plt.show()
    
    #################
    
        heatmap = np.zeros((100, 100))
        final_visited_maps = [agent.behavior for agent in population]

        for visited in final_visited_maps:
            for (x, y) in visited:
                heatmap[y][x] += 1
                
        for wall in farol.walls:
            heatmap[wall.y][wall.x] = np.nan
        plt.figure(figsize=(10, 10))
        # Create the heatmap on a specific axes object
        # Mask the walls so they appear as black
        mask = np.isnan(heatmap)

        ax = sns.heatmap(
            heatmap, 
            cmap="YlGnBu", 
            mask=mask,       # Mask walls
            cbar=True, 
            square=True
        )

        # Overlay walls as black squares
        for wall in farol.walls:
            ax.add_patch(plt.Rectangle(
                (wall.x, wall.y), 1, 1, color='black', lw=0
            ))
        
        tick_locations = np.arange(0, 101, 20)

        ax.set_xticks(tick_locations)
        ax.set_yticks(tick_locations)
        ax.set_xticklabels(tick_locations)
        ax.set_yticklabels(tick_locations)

        # Invert the y-axis to match the path plot
        # This puts y=99 (start) at the TOP-LEFT
        ax.invert_yaxis()
        ax.set_xlim(-1, 100)
        ax.set_ylim(-1, 100)
        # ---------------------------------
        plt.text(farol.goalx + 0.5, farol.goaly + 0.5, "Farol", color='black', fontsize=8, ha='center', va='center', fontweight='bold')
        # ----------------------------------------------

        

        plt.title(f"Heatmap of Visited Areas by Final Population (Gen {NUM_GENERATIONS})")
        plt.xlabel("X")
        plt.ylabel("Y")
        #plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(NUM_GENERATIONS), avg_fitness_per_gen, marker='o')
        plt.title("Average Combined Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Average Combined Fitness Score")
        plt.grid(True)
        
        plt.show()
        