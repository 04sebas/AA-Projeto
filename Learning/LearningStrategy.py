from abc import ABC, abstractmethod

class LearningStrategy(ABC):
    def __init__(self, nn_arch=(15, 4, (16, 8)), verbose=True):
        self.nn_arch = list(nn_arch)
        self.verbose = verbose
        self.fitness_history = []
        self.path_history = []
        self.best_nn = None
        self.best_weights = None

    @abstractmethod
    def choose_action(self, state, possible_actions):
        pass

    @abstractmethod
    def train(self, environment, training_positions=None):
        pass


    def generate_plots(self, environment, fitness_title="Fitness", paths_title="Paths", other_plots=None):
        if not self.verbose:
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import matplotlib.cm as cm
            import numpy as np

            width = getattr(environment, "width", 50)
            height = getattr(environment, "height", 50)

            raw_resources = getattr(environment, "resources", {})
            raw_nests = getattr(environment, "nests", [])
            raw_obstacles = getattr(environment, "obstacles", [])

            resources_list = []
            if isinstance(raw_resources, dict):
                for pos, info in raw_resources.items():
                    resources_list.append((tuple(pos), dict(info)))
            elif isinstance(raw_resources, (list, tuple)):
                for r in raw_resources:
                    if isinstance(r, dict) and "pos" in r:
                        resources_list.append(
                            (tuple(r["pos"]), {"quantity": r.get("quantity", 1), "value": r.get("value", 0)}))

            nests_list = []
            if isinstance(raw_nests, (list, tuple, set)):
                for n in raw_nests:
                    try:
                        nests_list.append(tuple(n))
                    except:
                        pass

            obstacles_list = []
            if isinstance(raw_obstacles, (list, tuple, set)):
                for o in raw_obstacles:
                    if isinstance(o, tuple) and len(o) >= 2:
                        obstacles_list.append(tuple(o))
                    elif isinstance(o, list) and len(o) >= 2:
                        obstacles_list.append(tuple(o))
                    elif isinstance(o, dict) and "pos" in o:
                        obstacles_list.append(tuple(o["pos"]))

            if self.fitness_history:
                plt.figure(figsize=(10, 4.5))
                plt.plot(range(len(self.fitness_history)), self.fitness_history, marker='o')
                plt.title(fitness_title)
                plt.xlabel("Iteration / Generation")
                plt.ylabel("Fitness / Reward")
                plt.grid(True)
                plt.tight_layout()

            if self.path_history:
                fig, ax = plt.subplots(figsize=(10, 10))

                for (rx, ry), info in resources_list:
                    ax.add_patch(
                        patches.Circle((rx, ry), radius=0.4, facecolor="gold", alpha=0.6, edgecolor='k', linewidth=0.3))
                    q = info.get("quantity", "")
                    ax.text(rx, ry, f"{q}", color="black", ha="center", va="center", fontsize=7)

                for nx, ny in nests_list:
                    ax.add_patch(patches.Circle((nx, ny), radius=0.5, facecolor="blue", edgecolor='k'))
                    ax.text(nx, ny, "N", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

                for ox, oy in obstacles_list:
                    ax.add_patch(patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black"))

                total_qty = len(self.path_history)
                top_n = min(5, total_qty)
                
                indices = list(range(total_qty))
                if len(self.fitness_history) == total_qty:
                    sorted_indices = sorted(indices, key=lambda i: self.fitness_history[i], reverse=True)
                    best_indices = sorted_indices[:top_n]
                else:
                    best_indices = indices[-top_n:]

                colors = cm.rainbow(np.linspace(0, 1, len(best_indices)))

                for idx_c, real_idx in enumerate(best_indices):
                    path = self.path_history[real_idx]
                    if not path: continue
                    xs = [p[0] for p in path]
                    ys = [p[1] for p in path]
                    fit_val = self.fitness_history[real_idx] if real_idx < len(self.fitness_history) else 0.0
                    ax.plot(xs, ys, label=f"Iter {real_idx} (Fit: {fit_val:.2f})", alpha=0.7, color=colors[idx_c])
                    ax.plot(xs[-1], ys[-1], 'x', markersize=8, color=colors[idx_c])

                ax.set_xlim(-1, width + 1)
                ax.set_ylim(-1, height + 1)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(paths_title)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.grid(True)
                if top_n > 0:
                    ax.legend(loc='upper right', fontsize='small')
                plt.tight_layout()

            if other_plots:
                other_plots(plt)

            plt.show()

        except Exception as e:
            print(f"[LearningStrategy] Error generating plots: {e}")
