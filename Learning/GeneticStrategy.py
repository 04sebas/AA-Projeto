import random
import numpy as np
from Agents.LearningAgent import LearningAgent
from Environments.LighthouseEnvironment import LighthouseEnvironment
from Objects.Action import Action
from Learning.NeuralNetwork import create_network_architecture
from Learning.LearningStrategy import LearningStrategy


class GeneticStrategy(LearningStrategy):
    def __init__(
        self,
        population_size=100,
        mutation_rate=0.01,
        num_generations=25,
        tournament_size=3,
        elitism_frac=0.2,
        nn_arch=(15, 4, (16, 8)),
        steps_per_evaluation=1000,
        mutation_std=0.1,
        trials_per_evaluation=3
    ):
        super().__init__(nn_arch=nn_arch, verbose=True)
        self.population_size = int(population_size)
        self.mutation_rate = float(mutation_rate)
        self.num_generations = int(num_generations)
        self.tournament_size = int(tournament_size)
        self.elitism_frac = float(elitism_frac)
        self.steps_per_evaluation = int(steps_per_evaluation)
        self.mutation_std = float(mutation_std)
        self.trials_per_evaluation = int(trials_per_evaluation)

        self.num_weights = None
        self.population = None
        self.fitness = None
        self.trained = False

    def choose_action(self, state, possible_actions):
        pass

    def _select_tournament(self):
        competitors = random.sample(range(self.population_size), self.tournament_size)
        best = max(competitors, key=lambda i: self.fitness[i])
        return best

    def _crossover(self, p1, p2):
        if self.num_weights <= 1:
            return p1.copy(), p2.copy()
        point = random.randint(1, self.num_weights - 1)
        child1 = np.concatenate([p1[:point], p2[point:]]).astype(np.float32)
        child2 = np.concatenate([p2[:point], p1[point:]]).astype(np.float32)
        return child1, child2

    def _mutate_weights(self, weights):
        w = weights.copy()
        mask = np.random.rand(self.num_weights) < self.mutation_rate
        if np.any(mask):
            w[mask] += np.random.randn(np.sum(mask)).astype(np.float32) * self.mutation_std
        return w

    def _initialize_population(self):
        nn_proto = create_network_architecture(*self.nn_arch)
        self.num_weights = int(nn_proto.calculate_number_weights())
        pop = []
        for _ in range(self.population_size):
            w = np.random.uniform(-1.0, 1.0, size=(self.num_weights,)).astype(np.float32)
            pop.append(w)
        self.population = pop
        self.fitness = np.zeros(self.population_size, dtype=np.float32)

    def train(self, environment, verbose=True, input_size=3, training_positions=None, sensor_range=3):
        self.verbose = verbose
        dummy_agent = LearningAgent(policy={"range": sensor_range})
        if hasattr(environment, "get_action_names"):
            dummy_agent.set_action_space(environment.get_action_names())
        num_actions = len(dummy_agent.action_names)
        available_actions = dummy_agent.action_names
        # sensor_range is now argument
        self.nn_arch = (input_size, num_actions, self.nn_arch[2] if len(self.nn_arch) > 2 else (16, 8))

        self._initialize_population()

        self.fitness_history = []
        self.path_history = []
        
        n_elite = max(1, int(self.elitism_frac * self.population_size))

        if self.verbose:
            print(f"[GA] Population: {self.population_size}, Generations: {self.num_generations}, Elite: {n_elite}, Trials/Ind: {self.trials_per_evaluation}")

        for gen in range(self.num_generations):
            if self.verbose:
                print(f"[GA] Generation {gen + 1}/{self.num_generations}")

            agents_per_generation = []

            for i, weights in enumerate(self.population):
                # Prepare Agent with weights
                nn = create_network_architecture(*self.nn_arch)
                nn.load_weights(weights)
                agent = LearningAgent(policy={"range": sensor_range}, action_names=available_actions)
                agent.neural_network = nn
                agent.weights = weights.copy()

                trial_fitnesses = []
                last_path = []

                for trial in range(self.trials_per_evaluation):
                    environment.restart()
                    environment.positions = {} # Ensure clean slate

                    agent.pos = None
                    agent.found_target = False
                    agent.total_reward = 0.0
                    agent.resources_collected = 0
                    agent.resources_deposited = 0

                    if isinstance(environment, LighthouseEnvironment):
                        if training_positions:
                            start = random.choice(training_positions)
                        else:
                            start = environment.random_position()
                    else:
                        start = (0, 0) # Default foraging check if needed

                    environment.positions[agent] = tuple(start)
                    agent.pos = tuple(start)
                    current_path = [agent.pos]
                    fit = 0.0

                    for step in range(self.steps_per_evaluation):
                        obs = environment.observation_for(agent)
                        agent.observation(obs)
                        acc = agent.act()

                        reward = environment.act(acc, agent) or None
                        fit += float(reward)
                        current_path.append(agent.pos)
                        finished_flag = environment.finished([agent])

                        if finished_flag:
                            break
                    
                    trial_fitnesses.append(fit)
                    last_path = current_path
                
                avg_fitness = sum(trial_fitnesses) / len(trial_fitnesses)
                
                # Adding manual attribute path to agent just for recording
                agent.path = last_path
                # We update agent reward to average for consistency if inspected later
                agent.total_reward = avg_fitness

                self.fitness[i] = avg_fitness
                agents_per_generation.append(agent)

            order = np.argsort(-self.fitness)
            self.population = [self.population[idx] for idx in order]
            self.fitness = self.fitness[order]
            agents_per_generation = [agents_per_generation[idx] for idx in order]

            best_fit = float(self.fitness[0])
            avg_fit = float(np.mean(self.fitness))
            
            # Update history for plots
            self.fitness_history.append(avg_fit)
            if agents_per_generation:
                 best_agent = agents_per_generation[0]
                 self.path_history.append(getattr(best_agent, "path", []))

            if self.verbose:
                print(f"[GA] Gen {gen + 1}: best={best_fit:.2f}, avg={avg_fit:.2f}")

            new_pop = []
            for k in range(n_elite):
                new_pop.append(self.population[k].copy())

            while len(new_pop) < self.population_size:
                p1_idx = self._select_tournament()
                p2_idx = self._select_tournament()
                p1 = self.population[p1_idx].copy()
                p2 = self.population[p2_idx].copy()

                child1, child2 = self._crossover(p1, p2)
                child1 = self._mutate_weights(child1)
                if len(new_pop) < self.population_size:
                    new_pop.append(child1)
                if len(new_pop) < self.population_size:
                    child2 = self._mutate_weights(child2)
                    new_pop.append(child2)

            self.population = new_pop
            self.fitness = np.zeros(self.population_size, dtype=np.float32)

        best_weights = self.population[0].copy()
        best_nn = create_network_architecture(*self.nn_arch)
        best_nn.load_weights(best_weights)

        self.best_weights = best_weights
        self.best_nn = best_nn
        self.trained = True

        self.generate_plots(environment, fitness_title="Average Fitness per Generation", paths_title="Evolution of Best Paths")

        return best_weights, best_nn
