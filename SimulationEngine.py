import json
import numpy as np

from matplotlib import pyplot as plt
from Agents.LearningAgent import LearningAgent
from Agents.FixedAgent import FixedAgent
from Environments.FarolEnvironment import FarolEnvironment
from Environments.ForagingEnvironment import ForagingEnvironment
from Learning.GeneticStrategy import GeneticStrategy
from Learning.QLearningStrategy import QLearningStrategy


def _save_best_nn(environment, agent_index, weights, nn_object, nn_arch=None, output_folder="models", type_str="genetic"):
    import os, pickle, re, time

    os.makedirs(output_folder, exist_ok=True)

    env_name = getattr(environment, "name", type(environment).__name__)
    base_prefix = f"{env_name}_agent{agent_index}_{type_str}"

    pattern = re.compile(rf"^{re.escape(base_prefix)}(?:_v(\d+))?\.pkl$")
    max_v = 0
    for fn in os.listdir(output_folder):
        m = pattern.match(fn)
        if m:
            if m.group(1):
                try:
                    v = int(m.group(1))
                    if v > max_v:
                        max_v = v
                except Exception:
                    pass
            else:
                if max_v < 1:
                    max_v = 1

    next_v = max_v + 1 if max_v > 0 else 1
    filename = f"{base_prefix}_v{next_v}.pkl"

    path = os.path.join(output_folder, filename)

    meta = {
        "env_name": env_name,
        "agent_index": agent_index,
        "nn_arch": nn_arch if nn_arch is not None else getattr(nn_object, "nn_arch", None),
        "type": type_str,
        "timestamp": int(time.time()),
    }

    data = {
        "meta": meta,
        "weights_flat": weights,
        "hidden_weights": getattr(nn_object, "hidden_weights", None),
        "hidden_biases": getattr(nn_object, "hidden_biases", None),
        "output_weights": getattr(nn_object, "output_weights", None),
        "output_bias": getattr(nn_object, "output_bias", None),
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"[SAVED] NN (type={type_str}) saved in {path}")
    return path, meta

class SimulationEngine:
    def __init__(self):
        self._config_file = None
        self.initial_resources = None
        self.environment = None
        self.steps = 0
        self.agents = []
        self.order = []
        self.active = False
        self.max_steps = 1000
        self.visualization = True
        self.agent_history = {}
        self.reward_history = {}
        self.strategy_configs = []
        self.training_positions = []
        self.test_positions = []

    def list_agents(self):
        return self.agents

    def create(self, param_filename):
        try:
            with open(param_filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._create_environment(config.get("environment", {}))
            self.strategy_configs = config.get("strategy", [])
            self._create_agents(config.get("agents", []))
            sim_config = config.get("simulator", {})
            self.max_steps = sim_config.get("max_steps", 100)
            self.visualization = sim_config.get("visualization", True)
            self.active = True
            self._config_file = param_filename
            
            self.training_positions, self.test_positions = self.generate_train_test_dataset()
            
            print(f"Simulation created successfully: {len(self.agents)} agents in environment")
        except FileNotFoundError:
            print(f"Error: File {param_filename} not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file: {param_filename}")
        except Exception as e:
            print(f"Error creating simulation: {e}")
        return self

    def _create_environment(self, env_config):
        env_type = env_config.get("type", "FarolEnvironment")

        if env_type == "FarolEnvironment" or env_type == "AmbienteFarol" or env_type == "LighthouseEnvironment":
            self.environment = FarolEnvironment(
                width=env_config.get("width", 100),
                height=env_config.get("height", 100),
                farol_pos=tuple(env_config.get("farol_pos", env_config.get("lighthouse_pos", [50, 75]))),
                obstacles=env_config.get("obstacles", [])
            )
        elif env_type == "ForagingEnvironment" or env_type == "AmbienteForaging":
            self.environment = ForagingEnvironment(
                width=env_config.get("width", 100),
                height=env_config.get("height", 100),
                resources=env_config.get("resources", []),
                nests=env_config.get("nests", []),
                obstacles=env_config.get("obstacles", [])
            )
        else:
            raise ValueError(f"Unknown environment type: {env_type}")

    def _get_strategy_config(self, strategy_type):
        for e in self.strategy_configs:
            if e.get("type") == strategy_type:
                return e
        return {}

    def _create_agents(self, agents_config):
        for config in agents_config:
            ag_type = config.get("type")
            quantity = config.get("quantity", 1)
            positions = config.get("initial_position", "random")
            trainable = config.get("trainable", True)

            for i in range(quantity):
                if positions == "random":
                    pos = self.environment.random_position()
                else:
                    if isinstance(positions, list) and i < len(positions):
                        pos = positions[i]
                    else:
                        pos = positions
                if isinstance(pos, list):
                    pos = tuple(pos)
                if ag_type == "FixedAgent" or ag_type == "AgenteFixo":
                    agent_policy = config.get("policy", {}) or {}
                    policy_type = agent_policy.get("type", "fixed")
                    agent_name = f"FixedAgent_{policy_type}_{len(self.agents)}"
                    agent = FixedAgent(
                        position=list(pos),
                        policy=agent_policy,
                        name=agent_name
                    )
                elif ag_type == "LearningAgent" or ag_type == "AgenteAprendizagem":
                    agent_policy = config.get("policy", {}) or {}
                    strategy_type = config.get("strategy_type", "genetic")
                    agent_name = f"LearningAgent_{strategy_type}_{len(self.agents)}"
                    agent = LearningAgent(
                        name=agent_name,
                        policy=agent_policy,
                        position=list(pos)
                    )
                    agent.trainable = bool(config.get("trainable", False))
                    agent.strategy_type = strategy_type
                    agent.strategy_conf = config.get("strategy_conf", {})
                else:
                    raise ValueError(f"Unknown agent type: {ag_type}")

                agent.trainable = bool(trainable)
                agent.pos = list(pos)
                self.agents.append(agent)
                print(f"Created agent {agent.name} at position {pos}")

    def load_network(self, file_path, agent_index):
        import pickle, os
        from Learning.NeuralNetwork import NeuralNetwork, relu, output_function

        base = os.path.dirname(__file__)
        full_path = os.path.join(base, file_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Not found: {full_path}")

        with open(full_path, "rb") as f:
            data = pickle.load(f)

        meta = data.get("meta", {})
        nn_arch = meta.get("nn_arch")
        type_str = meta.get("type", "unknown")

        nn_obj = None
        try:
            if nn_arch:
                input_size, output_size, hidden_arch = nn_arch
                nn_obj = NeuralNetwork(input_size, output_size, hidden_arch, relu, output_function)
                if data.get("hidden_weights") is not None:
                    nn_obj.hidden_weights = data.get("hidden_weights")
                if data.get("hidden_biases") is not None:
                    nn_obj.hidden_biases = data.get("hidden_biases")
                if data.get("output_weights") is not None:
                    nn_obj.output_weights = data.get("output_weights")
                if data.get("output_bias") is not None:
                    nn_obj.output_bias = data.get("output_bias")

                if hasattr(nn_obj, "weights") and data.get("weights_flat") is not None:
                    try:
                        nn_obj.weights = data.get("weights_flat")
                    except Exception:
                        pass

        except Exception as e:
            print(f"[LOAD] Warning: could not reconstruct NeuralNetwork from meta.nn_arch: {e}")
            nn_obj = None

        agent = self.agents[agent_index]

        if nn_obj is not None:
            agent.neural_network = nn_obj
            agent.weights = data.get("weights_flat")
            print(f"[LOAD] NN reconstructed and assigned to agent {agent_index} (type={type_str})")
            return

        agent.weights = data.get("weights_flat")
        try:
            existing_nn = getattr(agent, "neural_network", None)
            if existing_nn is not None:
                if data.get("hidden_weights") is not None:
                    setattr(existing_nn, "hidden_weights", data.get("hidden_weights"))
                if data.get("hidden_biases") is not None:
                    setattr(existing_nn, "hidden_biases", data.get("hidden_biases"))
                if data.get("output_weights") is not None:
                    setattr(existing_nn, "output_weights", data.get("output_weights"))
                if data.get("output_bias") is not None:
                    setattr(existing_nn, "output_bias", data.get("output_bias"))
                print(f"[LOAD] Weights assigned to existing neural_network of agent {agent_index}")
                return
        except Exception:
            pass

        print(f"[LOAD] Only weights_flat loaded for agent {agent_index} (type={type_str}).")

    def load_networks_summary(self, pattern: str = None, file_map: dict = None, agents: list = None, verbose: bool = True):
        summary = {}

        if agents is None:
            agents = list(range(len(self.agents)))

        file_map = file_map or {}

        for idx in agents:
            if idx < 0 or idx >= len(self.agents):
                summary[idx] = {"status": "error", "msg": "Invalid agent index"}
                if verbose:
                    print(f"[LOAD_NETWORKS] Invalid index: {idx}")
                continue

            agent = self.agents[idx]

            if not getattr(agent, "trainable", False):
                summary[idx] = {"status": "ignored", "msg": "agent not trainable"}
                if verbose:
                    print(f"[LOAD_NETWORKS] Agent {idx} (not trainable) - ignored")
                continue

            file_path = None
            if idx in file_map:
                file_path = file_map[idx]
            elif pattern is not None:
                try:
                    file_path = pattern.format(idx=idx)
                except Exception as e:
                    summary[idx] = {"status": "error", "msg": f"Error formatting pattern: {e}"}
                    if verbose:
                        print(f"[LOAD_NETWORKS] Error formatting pattern for agent {idx}: {e}")
                    continue
            else:
                summary[idx] = {"status": "ignored", "msg": "no pattern nor file_map provided"}
                if verbose:
                    print(f"[LOAD_NETWORKS] Agent {idx} - no file specified")
                continue

            try:
                self.load_network(file_path, idx)
                summary[idx] = {"status": "loaded", "msg": file_path}
                if verbose:
                    print(f"[LOAD_NETWORKS] Success: agent {idx} <- {file_path}")
            except FileNotFoundError:
                summary[idx] = {"status": "missing", "msg": file_path}
                if verbose:
                    print(f"[LOAD_NETWORKS] File not found for agent {idx}: {file_path}")
            except Exception as e:
                summary[idx] = {"status": "error", "msg": str(e)}
                if verbose:
                    print(f"[LOAD_NETWORKS] Error loading agent {idx} from {file_path}: {e}")

        return summary

    def autoload_if_flag(self, pattern: str = None, file_map: dict = None, agents: list = None, verbose: bool = True):
        if getattr(self, "autoload_models", False):
            return self.load_networks_summary(pattern=pattern, file_map=file_map, agents=agents, verbose=verbose)
        if verbose:
            print("[autoload_if_flag] autoload_models not activated; no file loaded.")
        return {}


    def generate_train_test_dataset(self, n_cases=None, train_ratio=0.7):
        import random
        if n_cases is None:
            n_cases = self.environment.width // 2
            
        dataset = set()
        attempts = 0
        max_attempts = n_cases * 100
        
        if isinstance(self.environment.resources, dict):
            resources_pos = list(self.environment.resources.keys())
        elif isinstance(self.environment.resources, list):
            resources_pos = [tuple(r["pos"]) for r in self.environment.resources]
            
        nests_pos = getattr(self.environment, "nests", [])
        
        if hasattr(self.environment, "farol_pos"):
            resources_pos.append(self.environment.farol_pos)

        while len(dataset) < n_cases and attempts < max_attempts:
            x = random.randint(0, self.environment.width - 1)
            y = random.randint(0, self.environment.height - 1)
            pos = (x, y)
            attempts += 1
            
            if pos in self.environment.obstacles:
                continue
                
            for rp in resources_pos:
                if abs(x - rp[0]) <= 2 and abs(y - rp[1]) <= 2:
                    too_close = True
                    break
            if too_close: continue
            
            for np in nests_pos:
                if abs(x - np[0]) <= 1 and abs(y - np[1]) <= 1:
                    too_close = True
                    break
            if too_close: continue
            
            dataset.add(pos)
            
        dataset_list = list(dataset)
        random.shuffle(dataset_list)
        
        n_train = int(len(dataset_list) * train_ratio)
        train_data = dataset_list[:n_train]
        test_data = dataset_list[n_train:]
        
        print("\n" + "="*50)
        print("[DATASET GENERATION]")
        print(f"Map Size: {self.environment.width}x{self.environment.height}")
        print(f"Target positions: {n_cases}")
        print(f"Generated positions: {len(dataset_list)}")
        print(f"Training positions ({len(train_data)}):")
        print(f"  {train_data}")
        print(f"Testing positions ({len(test_data)}):")
        print(f"  {test_data}")
        print("="*50 + "\n")
        
        return train_data, test_data

    def training_phase(self):
        saved_files = {}
        
        if not hasattr(self, "training_positions") or not self.training_positions:
            self.training_positions, self.test_positions = self.generate_train_test_dataset()
            
        training_positions = self.training_positions
        test_positions = self.test_positions
        
        try:
            with open("test_dataset.json", "w") as f:
                json_test_pos = [list(p) for p in test_positions]
                json.dump(json_test_pos, f)
            print("[DATASET] Test positions saved in test_dataset.json")
        except Exception as e:
            print(f"[DATASET] Error saving test dataset: {e}")

        for idx, agent in enumerate(self.agents):
            if not getattr(agent, "trainable", False):
                continue

            strat_type = getattr(agent, "strategy_type", None)
            conf = getattr(agent, "strategy_conf", None)
            if conf is None:
                conf = self._get_strategy_config(strat_type or "genetic")

            available_actions = ["up", "down", "right", "left"]
            if getattr(self.environment, "resources", None):
                if "collect" not in available_actions:
                    available_actions.append("collect")
            if getattr(self.environment, "nests", None):
                if "deposit" not in available_actions:
                    available_actions.append("deposit")

            if hasattr(agent, "set_action_space"):
                agent.set_action_space(available_actions)
            else:
                agent.action_names = available_actions

            try:
                input_size = agent.get_input_size()
            except Exception:
                sensing_range = getattr(agent.sensors, "sensing_range", 3)
                input_size = (2 * sensing_range + 1) ** 2 - 1 + 5

            output_size = len(available_actions)

            if strat_type == "genetic":
                ag = GeneticStrategy(
                    population_size=conf.get("population_size", 100),
                    mutation_rate=conf.get("mutation_rate", 0.01),
                    num_generations=conf.get("num_generations", 25),
                    tournament_size=conf.get("tournament_size", 2),
                    elitism_frac=conf.get("elitism_frac", 0.1),
                    nn_arch=(input_size, output_size, conf.get("hidden", (16, 8))),
                    steps_per_evaluation=conf.get("steps_per_evaluation", 750),
                    mutation_std=conf.get("mutation_std", 0.1),
                )

                self.environment.positions = {}

                sensing_range = getattr(agent.sensors, "sensing_range", 3)
                best_weights, best_nn = ag.train(self.environment, verbose=True, input_size=input_size, training_positions=training_positions, sensor_range=sensing_range)

                self.environment.restart()

                agent.neural_network = best_nn
                agent.weights = np.array(best_weights).copy()

                path, meta = _save_best_nn(
                    self.environment,
                    idx,
                    best_weights,
                    best_nn,
                    nn_arch=getattr(ag, "nn_arch", None),
                    type_str="genetic"
                )

                meta = dict(meta or {})
                meta["type"] = "genetic"
                saved_files[idx] = {"path": path, "meta": meta}
                print(f"[training_phase] Agent {idx}: genetic strategy applied and saved to {path}")

            elif strat_type == "dqn":
                conf_dqn = conf or {}
                dqn = QLearningStrategy(
                    nn_arch=(input_size, output_size, conf_dqn.get("hidden", (16, 8))),
                    episodes=conf_dqn.get("episodes", 100),
                    steps_per_ep=conf_dqn.get("steps_per_ep", 750),
                    gamma=conf_dqn.get("gamma", 0.99),
                    epsilon=conf_dqn.get("epsilon", 0.90),
                    epsilon_min=conf_dqn.get("epsilon_min", 0.1),
                    epsilon_decay=conf_dqn.get("epsilon_decay", 0.95),
                    batch_size=conf_dqn.get("batch_size", 32),
                    target_update_freq=conf_dqn.get("target_update_freq", 50),
                    memory_size=conf_dqn.get("memory_size", 50000),
                    learning_rate=conf_dqn.get("learning_rate", 0.001),
                )

                self.environment.positions = {}

                best_weights, best_nn = dqn.train(self.environment, verbose=True, range_val=agent.sensors.sensing_range, training_positions=training_positions)

                self.environment.restart()

                agent.neural_network = best_nn
                agent.weights = np.array(best_weights).copy()

                path, meta = _save_best_nn(
                    self.environment,
                    idx,
                    best_weights,
                    best_nn,
                    nn_arch=getattr(dqn, "nn_arch", None),
                    type_str="dqn"
                )
                meta = dict(meta or {})
                meta["type"] = "dqn"
                saved_files[idx] = {"path": path, "meta": meta}
                print(f"[training_phase] Agent {idx}: dqn trained applied and saved to {path}")

            else:
                continue

        return saved_files

    def testing_phase(self):
        test_positions = getattr(self, "test_positions", None)
        
        if not test_positions:
                    if isinstance(test_positions, list):
                        test_positions = [tuple(p) for p in test_positions]
                    print(f"[TESTING_PHASE] Loaded {len(test_positions)} test positions from file.")
                except Exception as e:
                    print(f"[TESTING_PHASE] Error reading test_dataset.json: {e}")
        
        if test_positions:
            print("[TESTING_PHASE] Starting validation on test set (30%)...")
            
            fitness_results = []
            
            total_fitness = 0
            count = 0
            
            for pos in test_positions:
                self.environment.restart()
                self.environment.positions = {}
                
                for agent in self.agents:
                    agent.pos = list(pos)
                    agent.total_reward = 0.0
                    agent.found_target = False
                    agent.resources_collected = 0
                    agent.resources_deposited = 0
                    self.environment.positions[agent] = tuple(pos)
                
                self.run(max_steps=self.max_steps)
                
                current_fit = 0
                for agent in self.agents:
                    current_fit += getattr(agent, "total_reward", 0.0)
                
                fitness_results.append(current_fit)
                total_fitness += current_fit
                count += 1
                
            avg_fit = total_fitness / count if count > 0 else 0
            print(f"[TESTING_PHASE] Average Fitness on Test: {avg_fit:.2f} (in {count} cases)")
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(fitness_results, marker='o')
            plt.title(f"Performance on Test Set (Average: {avg_fit:.2f})")
            plt.xlabel("Test Case")
            plt.ylabel("Fitness")
            plt.grid(True)
            plt.show()
            
        else:
            self.run(self.max_steps)

    def run(self, max_steps=None):
        if not self.active:
            print("Simulation not created correctly.")
            return

        max_s = max_steps if max_steps is not None else self.max_steps
        self.environment.positions = {}

        for agent in self.agents:
            if not hasattr(agent, "pos") or agent.pos is None:
                pos = self.environment.random_position()
                agent.pos = list(pos)
            self.environment.positions[agent] = tuple(agent.pos)

        self.agent_history = {}
        self.reward_history = {}

        for agent in self.agents:
            self.agent_history[agent] = [list(agent.pos)]
            name = getattr(agent, "name", f"Agent_{self.agents.index(agent)}")
            self.reward_history[name] = [0]
        self.initial_resources = {}

        if isinstance(self.environment.resources, dict):
            for pos, info in self.environment.resources.items():
                self.initial_resources[pos] = {"value": info.get("value", 0), "quantity": info.get("quantity", 0)}

        for step in range(max_s):
            all_finished = False
            for agent in self.agents:
                observation = self.environment.observation_for(agent)
                agent.observation(observation)
                action = agent.act()
                reward = self.environment.act(action, agent)
                agent.evaluate_current_state(reward)

                if isinstance(agent.pos, tuple):
                    agent.pos = list(agent.pos)
                self.agent_history[agent].append(list(agent.pos))
                name = getattr(agent, "name", f"Agent_{self.agents.index(agent)}")
                self.reward_history[name].append(reward)

            if hasattr(self.environment, "update"):
                self.environment.update()
            self.steps += 1

            if hasattr(self.environment, "finished"):
                all_finished = self.environment.finished(self.agents)
            else:
                 if isinstance(self.environment, FarolEnvironment):
                    all_finished = all(getattr(ag, "found_target", False) for ag in self.agents)
                 elif isinstance(self.environment, ForagingEnvironment):
                    all_finished = self.environment.finished(self.agents)

            if all_finished:
                print(f"Simulation finished at step {step}!")
                break

        if self.visualization:
            try:
                self.plot_test_statistics()
                self.visualize_paths()
            except Exception as e:
                print(f"Error drawing visualizations: {e}")

    def visualize_paths(self):
        height = self.environment.height
        width = self.environment.width
        plt.figure(figsize=(10, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(self.agent_history))))
        for i, (agent, path) in enumerate(self.agent_history.items()):
            if not path:
                continue
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            plt.plot(xs, ys, marker='.', linewidth=1, color=colors[i], label=f"{getattr(agent, 'name', f'Agent {i}')}")
            plt.scatter(xs[0], ys[0], c=[colors[i]], marker='o', s=80) 
            plt.scatter(xs[-1], ys[-1], c=[colors[i]], marker='x', s=80) 

            else:
                for r in resources:
                    pos = tuple(r.get("pos")) if isinstance(r.get("pos"), (list, tuple)) else None
                    if pos is not None:
                        resources_items.append((pos, r))

        if resources_items:
            values = [info.get("value", info.get("quantity", 1)) for (_p, info) in resources_items]
            quantities = [info.get("quantity", 1) for (_p, info) in resources_items]
            min_val, max_val = min(values), max(values)
            span = max(1e-6, max_val - min_val)
            cmap = plt.get_cmap("RdYlBu")
            for (pos, info), val, q in zip(resources_items, values, quantities):
                x, y = pos
                size = max(40, 40 + (q - min(quantities)) * 4)
                norm = (val - min_val) / span
                color = cmap(norm)
                plt.scatter([x], [y], s=size, marker='o', color=color, alpha=0.8, edgecolors='k')
                plt.text(x + 0.2, y + 0.2, f"q:{q}\nv:{val}", fontsize=7)

        nests = getattr(self.environment, "nests", None)
        if nests:
            nx = [p[0] for p in nests]
            ny = [p[1] for p in nests]
            plt.scatter(nx, ny, s=200, marker='^', color='gold', edgecolors='k', label="Nests")

        obs_raw = getattr(self.environment, "raw_obstacles", None)
        if obs_raw:
            obs_x = [p["pos"][0] for p in obs_raw]
            obs_y = [p["pos"][1] for p in obs_raw]
        else:
            obs_set = getattr(self.environment, "obstacles", None)
            if obs_set:
                obs_x = [p[0] for p in obs_set]
                obs_y = [p[1] for p in obs_set]
            else:
                obs_x, obs_y = [], []
        if obs_x:
            plt.scatter(obs_x, obs_y, color="black", marker='s', s=120, label="Obstacles")

        if hasattr(self.environment, "farol_pos"):
            farol_x, farol_y = self.environment.farol_pos
            plt.scatter([farol_x], [farol_y], color="red", marker="*", s=300, label="Farol")

        ax = plt.gca()
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal', adjustable='box')

        xticks = np.arange(0, width, max(1, width // 10))
        yticks = np.arange(0, height, max(1, height // 10))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Agent Trajectories in Environment")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_test_statistics(self):
        import matplotlib.pyplot as plt
        import numpy as np

        agents = self.agents
        names = [getattr(ag, "name", f"Agent_{i}") for i, ag in enumerate(agents)]

        collected = [getattr(ag, "resources_collected", 0) for ag in agents]
        deposited = [getattr(ag, "resources_deposited", 0) for ag in agents]
        fitnesses = [getattr(ag, "total_reward", 0.0) for ag in agents]

        x = np.arange(len(names))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, collected, width, label="Collected")
        plt.bar(x + width / 2, deposited, width, label="Deposited")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel("Quantity")
        plt.title("Resources Collected and Deposited by Agent (Test)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(x, fitnesses, width * 1.2)
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel("Fitness (Total Reward)")
        plt.title("Agent Fitness in Test Phase")
        plt.tight_layout()
        plt.show()

        print("\nTest Phase Summary:")
        print(f"{'Agent':20s} {'Collected':>10s} {'Deposited':>12s} {'Fitness':>12s}")
        for name, r, d, f in zip(names, collected, deposited, fitnesses):
            print(f"{name:20s} {r:10d} {d:12d} {f:12.2f}")

    def save_animation_gif(self, filepath, fps=10, trail_len=20):
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter

        if not self.agent_history:
            raise RuntimeError("agent_history empty")

        agents = list(self.agent_history.keys())
        paths = [self.agent_history[ag] for ag in agents]
        max_steps = max(len(p) for p in paths)

        fig, ax = plt.subplots(figsize=(8, 8))
        width, height = self.environment.width, self.environment.height
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect("equal")
        ax.grid(True)

        resources = getattr(self.environment, "resources", {})
        if isinstance(resources, dict):
            for (x, y), info in resources.items():
                ax.scatter(x, y, s=80, color="gold", edgecolors="k")

        nests = getattr(self.environment, "nests", [])
        if nests:
            nx, ny = zip(*nests)
            ax.scatter(nx, ny, s=160, marker="^", color="blue", edgecolors="k")

        colors = plt.cm.tab20(np.linspace(0, 1, len(agents)))
        agents_scatter = []
        trails = []

        for c in colors:
            sc = ax.scatter([], [], s=80, color=c, edgecolors="k")
            ln, = ax.plot([], [], linewidth=1.5, color=c, alpha=0.7)
            agents_scatter.append(sc)
            trails.append(ln)

        def update(frame):
            for i, path in enumerate(paths):
                if frame < len(path):
                    agents_scatter[i].set_offsets([path[frame]])
                else:
                    agents_scatter[i].set_offsets([])

                start = max(0, frame - trail_len)
                seg = path[start:frame + 1]
                if len(seg) > 1:
                    xs, ys = zip(*seg)
                    trails[i].set_data(xs, ys)
                else:
                    trails[i].set_data([], [])

            ax.set_title(f"Step {frame}")
            return agents_scatter + trails

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=max_steps,
            interval=1000 // fps,
            blit=False
        )

        anim.save(filepath, writer=PillowWriter(fps=fps))
        plt.close(fig)

        print(f"[GIF] Animação salva em {filepath}")

    def run_experiments(self, num_runs: int = 30, max_steps: int = None, file_map: dict = None, agents_to_load: list = None, seed: int = None, save_plot: str = None):
        import numpy as _np
        import random

        if file_map:
            try:
                self.load_networks_summary(file_map=file_map, agents=agents_to_load or list(file_map.keys()))
            except Exception as e:
                print(f"[run_experiments] Warning loading networks: {e}")

        visual_original = getattr(self, "visualization", False)
        self.visualization = False

        agent_names = [getattr(a, "name", f"Agent_{i}") for i, a in enumerate(self.agents)]
        fitness_history = []

        is_foraging = "ForagingEnvironment" in getattr(self.environment, "name", "") or "Foraging" in str(type(self.environment))

        if is_foraging:
            resources_per_agent = {name: {"collected": 0, "deposited": 0} for name in agent_names}
        else:
            successes_per_agent = {name: 0 for name in agent_names}

        for i in range(num_runs):
            if seed is not None:
                s = int(seed) + i
                _np.random.seed(s)
                random.seed(s)

            if hasattr(self.environment, "restart"):
                self.environment.restart()
            elif hasattr(self.environment, "reset"):
                self.environment.reset()
                
            self.environment.positions = {}

            for agent in self.agents:
                agent.pos = None
                agent.total_reward = 0.0
                agent.found_target = False

                if hasattr(agent, "resources_collected"):
                    agent.resources_collected = 0
                if hasattr(agent, "resources_deposited"):
                    agent.resources_deposited = 0

            self.run(max_steps=max_steps)

            total_fitness = 0.0
            for idx, agent in enumerate(self.agents):
                name = getattr(agent, "name", f"Agent_{idx}")
                total_fitness += float(getattr(agent, "total_reward", 0.0) or 0.0)
                if is_foraging:
                    collected = int(getattr(agent, "resources_collected", 0) or 0)
                    deposited = int(getattr(agent, "resources_deposited", 0) or 0)
                    resources_per_agent[name]["collected"] += collected
                    resources_per_agent[name]["deposited"] += deposited
                else:
                    if bool(getattr(agent, "found_target", False)):
                        successes_per_agent[name] += 1

            fitness_history.append(total_fitness)

            try:
                self.environment.positions = {}
            except Exception:
                pass

        self.visualization = visual_original

        fitness_array = _np.array(fitness_history, dtype=float)
        stats = {
            "mean_fitness": float(fitness_array.mean()) if len(fitness_array) else 0.0,
            "std_fitness": float(fitness_array.std()) if len(fitness_array) else 0.0
        }

        if is_foraging:
            stats["total_resources"] = {name: {"collected": v["collected"], "deposited": v["deposited"]} for name, v in resources_per_agent.items()}
        else:
            stats["successes_per_agent"] = successes_per_agent

        try:
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker="o")
            plt.xlabel("Simulation")
            plt.ylabel("Total Fitness")
            plt.title(f"Fitness per simulation (mean={stats['mean_fitness']:.2f}, std={stats['std_fitness']:.2f})")
            plt.grid(True)
            plt.tight_layout()
            if save_plot:
                plt.savefig(save_plot.replace(".png", "_fitness.png"), dpi=200)
            plt.show()
        except Exception as e:
            print(f"[run_experiments] Error drawing fitness plot: {e}")

        try:
            plt.figure(figsize=(10, 4))
            names = agent_names
            if is_foraging:
                collected = [resources_per_agent[n]["collected"] for n in names]
                deposited = [resources_per_agent[n]["deposited"] for n in names]
                import numpy as _np_local
                x = _np_local.arange(len(names))
                width = 0.35
                plt.bar(x - width / 2, collected, width, label="Collected")
                plt.bar(x + width / 2, deposited, width, label="Deposited")
                plt.xticks(x, names, rotation=45, ha="right")
                plt.ylabel("Resources (total over all simulations)")
                plt.title("Resources collected and deposited by agent (Total)")
                plt.legend()
                plt.tight_layout()
                if save_plot:
                    plt.savefig(save_plot.replace(".png", "_resources.png"), dpi=200)
                plt.show()
            else:
                successes = [successes_per_agent[n] for n in names]
                plt.bar(names, successes)
                plt.xlabel("Agent")
                plt.ylabel("No. of Successes")
                plt.title("Successes by Agent (Total)")
                plt.tight_layout()
                if save_plot:
                    plt.savefig(save_plot.replace(".png", "_successes.png"), dpi=200)
                plt.show()

        except Exception as e:
            print(f"[run_experiments] Error drawing bar plot: {e}")

if __name__ == "__main__":
    simulator = SimulationEngine().create("SimuladorFarolVazio.json")
    if simulator.active:
        simulator.training_phase()
        simulator.testing_phase()
        simulator.save_animation_gif("models/trajectories_foraging.gif", fps=12, trail_len=30)
