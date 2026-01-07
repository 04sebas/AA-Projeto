import math
import random
import numpy as np
from Agents.Agent import Agent
from Objects.Action import Action

class LearningAgent(Agent):
    def __init__(self, name="LA", policy=None, position=None, action_names=None):
        position = position if position is not None else [0, 0]
        policy = policy or {}
        if "range" not in policy:
            policy["range"] = 1
            
        super().__init__(position, name, policy)
        
        self.neural_network = None
        self.weights = None
        self.trainable = True
        
        if action_names is None:
            self.action_names = ["up", "down", "right", "left", "collect", "deposit"]
        else:
            self.action_names = list(action_names)

        self.strategy_type = self.policy.get("strategy_type", "genetic")
        self.strategy_conf = self.policy.get("strategy_conf", {})
        
        self.known_resources = set()
        self.known_nests = set()
        self.last_resource_value = 0

    def observation(self, obs):
        super().observation(obs)
        current_pos = tuple(getattr(obs, "position", None))
        perceptions = getattr(obs, "perceptions", []) or []

        for p in perceptions:
            p_type = p.get("type")
            pos_t = tuple(p.get("pos"))

            if p_type in ("resource", "beacon"):
                quantity = p.get("quantity", 0)
                if quantity and quantity > 0:
                    self.known_resources.add(pos_t)
                else:
                    self.known_resources.discard(pos_t)

            elif p_type == "nest":
                self.known_nests.add(pos_t)

            elif p_type == "agent":
                from_agent = p.get("ref")
                if from_agent is not None:
                    self.communicate("foraging", from_agent)

        if current_pos in self.known_resources:
            if not any(tuple(p.get("pos")) == current_pos and p.get("type") in ("resource", "beacon") for p in perceptions):
                self.known_resources.discard(current_pos)

    def create(self, param_file):
        return self

    def act(self):
        if self.neural_network is None or self.last_obs is None:
            return Action(random.choice(self.action_names[:4]))

        obs = self.last_obs
        px, py = obs.position
        perceptions = obs.perceptions or []

        if any(tuple(p.get("pos")) == (px, py) and p.get("type") in ("resource", "beacon") for p in perceptions):
            if not getattr(obs, "load", 0) > 0:
                return Action("collect")

        if any(tuple(p.get("pos")) == (px, py) and p.get("type") == "nest" for p in perceptions):
            if getattr(obs, "load", 0) > 0:
                return Action("deposit")

        if getattr(obs, "load", 0) > 0:
            visible_nests = [tuple(p.get("pos")) for p in perceptions if p.get("type") == "nest"]
            if visible_nests:
                target = min(visible_nests, key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
                return Action(self.__action_move_to(target, (px, py)))

        visible_agents = [p.get("ref") for p in perceptions if p.get("type") == "agent" and p.get("ref") is not None]
        if visible_agents:
            for from_ag in visible_agents:
                self.communicate("foraging", from_ag)

        if getattr(obs, "load", 0) == 0:
            visible_resources = [tuple(p.get("pos")) for p in perceptions if p.get("type") in ("resource", "beacon")]
            if visible_resources:
                target = min(visible_resources, key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
                return Action(self.__action_move_to(target, (px, py)))

        goal = getattr(obs, "goal", None) 
        if goal is None:
            return Action(random.choice(self.action_names[:4]))

        nn_input = self.build_nn_input(self.last_obs)

        if self.strategy_type in ("dqn", "genetic"):
            output = self.neural_network.propagate(nn_input)
            action_idx = int(np.argmax(output))
        else:
            return Action("stay")

        action_idx = max(0, min(action_idx, len(self.action_names) - 1))
        return Action(self.action_names[action_idx])

    def communicate(self, message, from_agent):
        if message == "beacon":
            return

        if message == "foraging":
            resources_other = getattr(from_agent, "known_resources", None)
            nests_other = getattr(from_agent, "known_nests", None)

            if isinstance(resources_other, set):
                self.known_resources.update(resources_other)

            if isinstance(nests_other, set):
                self.known_nests.update(nests_other)
            return

    def neighborhood(self):
        obs = self.last_obs
        if not obs:
            input_size = (2 * self.sensors.sensing_range + 1) ** 2 - 1
            return [-0.9] * input_size

        px, py = obs.position
        sensing_range = self.sensors.sensing_range
        perceptions = obs.perceptions or []
        features = []

        goal = getattr(obs, "goal", None)
        if goal is not None:
            gx, gy = goal
            current_dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        else:
            gx = gy = None
            current_dist = None

        for dy in range(-sensing_range, sensing_range + 1):
            for dx in range(-sensing_range, sensing_range + 1):
                if dx == 0 and dy == 0:
                    continue

                check_pos = (px + dx, py + dy)
                obj = next((p for p in perceptions if tuple(p["pos"]) == check_pos), None)

                if obj:
                    obj_type = obj.get("type", "")
                    if obj_type == "obstacle":
                        features.append(-0.9)
                    elif obj_type in ("resource", "beacon"):
                        if getattr(obs, "load", 0) <= 0:
                            features.append(0.9)
                        else:
                            features.append(0.1)
                    elif obj_type == "nest":
                        if getattr(obs, "load", 0) > 0:
                            features.append(1.0)
                        else:
                            features.append(0.1)
                    else:
                        features.append(0.0)
                else:
                    if gx is not None and gy is not None:
                        cell_dist = math.sqrt((check_pos[0] - gx) ** 2 + (check_pos[1] - gy) ** 2)
                        if cell_dist < current_dist:
                            features.append(0.9)
                        else:
                            features.append(-0.9)
                    else:
                        features.append(-0.9)

        return features

    def build_nn_input(self, obs):
        px, py = obs.position
        features = np.array(self.neighborhood(), dtype=np.float32)

        width = max(1.0, getattr(obs, "width", 1))
        height = max(1.0, getattr(obs, "height", 1))

        norm_x = px / width
        norm_y = py / height

        goal = getattr(obs, "goal", None)
        if goal is not None:
            gx, gy = goal
            obj_x = (gx - px) / width
            obj_y = (gy - py) / height
        else:
            obj_x = 0.0
            obj_y = 0.0

        loading = float(getattr(obs, "load", 0) > 0)

        return np.concatenate(([norm_x, norm_y], features, [obj_x, obj_y, loading])).astype(np.float32)

    def get_input_size(self):
        sensing_range = self.sensors.sensing_range
        num_features = (2 * sensing_range + 1) ** 2 - 1
        return int(num_features + 5)

    def set_action_space(self, action_names):
        self.action_names = list(action_names)

    def __action_move_to(self, target, current):
        tx, ty = target
        cx, cy = current
        dx = tx - cx
        dy = ty - cy

        if abs(dx) > abs(dy):
            if dx > 0:
                return "right"
            elif dx < 0:
                return "left"
        else:
            if dy > 0:
                return "down"
            elif dy < 0:
                return "up"
        return "stay"
