import math
from copy import deepcopy
from ProjetoAA.Agents.Agent import Agent
from ProjetoAA.Environments.Environment import Environment
from ProjetoAA.Learning.Policies import DIRECTIONS
from ProjetoAA.Objects.Action import Action
from ProjetoAA.Objects.Observation import Observation

class ForagingEnvironment(Environment):
    def __init__(self, width=50, height=50, resources=None, nests=None, obstacles=None):
        self.initial_resources = {
            tuple(r["pos"]): {"value": r["valor"], "quantity": r["quantidade"]}
            for r in (resources or [])
        }
        super().__init__(width, height, resources=deepcopy(self.initial_resources), obstacles=obstacles, name="ForagingEnvironment")
        self.capacity = 1
        self.nests = [tuple(n) for n in (nests or [])]
        self.loads = {}
        self.targets = {}

    def observation_for(self, agent):
        pos = self.positions.get(agent, (0, 0))
        perceptions = agent.sensors.perceive(self, pos)
        
        target = self.targets.get(agent)
        if target and target not in self.resources and target not in self.nests:
            self.targets[agent] = None

        if pos in self.resources:
            r = self.resources[pos]
            qty = r.get("quantity", 0)
            if qty > 0:
                perceptions.append({"pos": pos, "type": "resource", "value": r.get("value", 1), "quantity": qty})

        if pos in self.nests:
            perceptions.append({"pos": pos, "type": "nest"})

        range_val = getattr(agent.sensors, "sensing_range", 3)
        for other, other_pos in self.positions.items():
            if other is agent: continue
            if abs(other_pos[0] - pos[0]) <= range_val and abs(other_pos[1] - pos[1]) <= range_val:
                perceptions.append({"pos": tuple(other_pos), "type": "agent", "ref": other})

        if not self.targets.get(agent):
            has_load = self.loads.get(agent, 0) > 0
            target = self._closest_nest(pos) if has_load else self._closest_resource(pos)
            if target: self.targets[agent] = target

        obs = Observation(perceptions)
        obs.position = pos
        obs.load = self.loads.get(agent, 0)
        obs.width = self.width
        obs.height = self.height
        obs.goal = self.targets.get(agent)
        agent.last_obs = obs
        obs.foraging = True
        return obs

    def act(self, action: Action, agent: Agent):
        pos = list(self.positions.get(agent, (0, 0)))
        dx, dy = DIRECTIONS.get(action.name, (0, 0))
        new_pos = (pos[0] + dx, pos[1] + dy)

        if not self.is_valid_position(*new_pos):
            return -1

        current_pos = tuple(pos)
        target = self.targets.get(agent)
        has_load = self.loads.get(agent, 0) > 0
        
        if not target:
            target = self._closest_nest(current_pos) if has_load else self._closest_resource(current_pos)
            if target: self.targets[agent] = target

        old_dist = self._normalized_distance(current_pos, target)
        self.positions[agent] = new_pos
        agent.pos = list(new_pos)
        
        if action.name == "collect" and new_pos in self.resources:
            load = self.loads.get(agent, 0)
            if load < self.capacity:
                resource = self.resources[new_pos]
                reward = 100.0 + resource.get("value", 25)
                # Assuming agent has this attribute logic, if not it will be ignored on agent side if not used. 
                # But agent classes were updated to use last_resource_value
                agent.last_resource_value = int(resource.get("value", 1))
                self.loads[agent] = load + 1
                
                if hasattr(agent, "resources_collected"): agent.resources_collected += 1
                
                resource["quantity"] = max(0, int(resource.get("quantity", 1)) - 1)
                if resource["quantity"] <= 0:
                    self._invalidate_targets_for_resource(new_pos)

                self.targets[agent] = self._closest_nest(new_pos) if self.loads[agent] >= self.capacity else (self._closest_resource(new_pos) or self._closest_nest(new_pos))
                return reward
            else:
                return 1.0
        
        elif action.name == "deposit" and new_pos in self.nests:
            load = self.loads.get(agent, 0)
            if load > 0:
                reward = 200.0 + float(getattr(agent, "last_resource_value", 1))
                self.loads[agent] = 0
                agent.last_resource_value = 0
                if hasattr(agent, "resources_deposited"): agent.resources_deposited += load
                self.targets[agent] = self._closest_resource(new_pos)
                return reward
            else:
                return 1.0

        elif action.name == "stay":
            return -0.3

        target_after = self.targets.get(agent)
        new_dist = self._normalized_distance(new_pos, target_after)
        
        reward = -0.1
        if old_dist is not None and new_dist is not None:
             if new_dist < old_dist: reward += 0.5
             elif new_dist > old_dist: reward -= 0.05
        return reward

    def restart(self):
        super().restart()
        self.loads = {}
        self.targets = {}
        self.resources = deepcopy(self.initial_resources)

    def _closest_resource(self, pos):
        if not self.resources: return None
        best, best_d = None, None
        for rpos, info in self.resources.items():
            if info.get("quantity", 0) <= 0: continue
            d = math.hypot(rpos[0]-pos[0], rpos[1]-pos[1])
            if best_d is None or d < best_d:
                best_d = d
                best = rpos
        return best

    def _closest_nest(self, pos):
        if not self.nests: return None
        best, best_d = None, None
        for npos in self.nests:
            d = math.hypot(npos[0]-pos[0], npos[1]-pos[1])
            if best_d is None or d < best_d:
                best_d = d
                best = npos
        return best

    def _normalized_distance(self, pos, target):
        if not target: return None
        dist = math.hypot(pos[0]-target[0], pos[1]-target[1])
        max_dist = math.hypot(self.width, self.height) or 1.0
        return dist / max_dist

    def _invalidate_targets_for_resource(self, resource_pos):
        for ag, tgt in list(self.targets.items()):
            if tgt == resource_pos:
                agent_pos = self.positions.get(ag)
                self.targets[ag] = self._closest_nest(agent_pos) if (agent_pos and self.loads.get(ag, 0) > 0) else self._closest_resource(agent_pos)

    def finished(self, agents=None) -> bool:
        resources_depleted = all(info.get("quantity", 0) <= 0 for info in self.resources.values())
        agents_to_check = agents if agents is not None else list(self.positions.keys())
        no_agent_carrying = all(self.loads.get(ag, 0) == 0 for ag in agents_to_check)
        return resources_depleted and no_agent_carrying

