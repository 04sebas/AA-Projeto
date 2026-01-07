import json
from Agents.Agent import Agent
from Learning.Policies import DIRECTIONS, random_policy, greedy_policy
from Objects.Action import Action
from Objects.Observation import Observation

class FixedAgent(Agent):
    def __init__(self, position=None, policy=None, name="FA"):
        position = position if position is not None else [0, 0]
        policy = policy or {}
        if "range" not in policy:
            policy["range"] = 5
            
        super().__init__(position, name, policy)
        
        self.trainable = False
        self.last_direction = None
        self.last_nest = None
        self.prev_position = None
        self.steps_without_move = 0
        self.known_resources = {}
        self.last_resource = None
        self.known_obstacles = set()
        self.stuck_threshold = self.policy.get("stuck_threshold", 2)

    @classmethod
    def create(cls, param_filename: str) -> "FixedAgent":
        try:
            with open(param_filename, 'r', encoding='utf-8') as f:
                config = json.load(f)

            agents_config = config.get("agents", [])

            for conf in agents_config:
                if conf.get("type") == "FixedAgent":
                    real_policy = conf.get("policy", {})

                    return cls(
                        policy=real_policy
                    )

            return cls()
        except Exception as e:
            print(f"Error creating FixedAgent from file: {e}")
            return cls()

    def observation(self, obs: Observation):
        super().observation(obs)

        current_pos = tuple(getattr(obs, "position", None))
        if self.prev_position is None:
            self.steps_without_move = 0
        else:
            if current_pos == self.prev_position:
                self.steps_without_move += 1
            else:
                self.steps_without_move = 0
        self.prev_position = current_pos

        perceptions = getattr(obs, "perceptions", []) or []
        for p in perceptions:
            p_type = p.get("type")
            if p_type in ("resource", "beacon"):
                pos_t = tuple(p.get("pos"))
                quantity = p.get("quantity", p.get("value", 1))
                if quantity and quantity > 0:
                    self.known_resources[pos_t] = quantity
                else:
                    self.known_resources.pop(pos_t, None)
            if p_type == "nest":
                self.last_nest = tuple(p.get("pos"))

        if current_pos in self.known_resources:
            if not any(tuple(p.get("pos")) == current_pos and p.get("type") in ("resource", "beacon")
                       for p in perceptions):
                self.known_resources.pop(current_pos, None)
                if self.last_resource == current_pos:
                    self.last_resource = None

        if self.steps_without_move >= self.stuck_threshold and self.last_direction:
            delta = DIRECTIONS.get(self.last_direction, (0, 0))
            if current_pos is not None:
                attempt = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                percep_pos = {tuple(p.get("pos")): p.get("type") for p in perceptions}
                front_type = percep_pos.get(attempt)
                if front_type not in ("resource", "beacon", "nest"):
                    self.known_obstacles.add(attempt)

    def act(self) -> Action:
        if self.found_target:
            return Action("stay")

        obs = self.last_obs
        if obs is None:
            return Action("stay")

        pos = tuple(self.pos)
        perceptions = getattr(obs, "perceptions", []) or []

        for p in perceptions:
            if tuple(p.get("pos")) == pos and p.get("type") == "nest":
                self.last_nest = pos
                if getattr(obs, "load", 0) > 0:
                    return Action("deposit")
            elif tuple(p.get("pos")) == pos:
                p_type = p.get("type")
                if p_type in ("resource", "beacon") and getattr(obs, "load", 0) == 0:
                    self.last_resource = pos
                    quantity = p.get("quantity", p.get("value", 1))
                    if quantity and quantity > 0:
                        self.known_resources[pos] = quantity
                    else:
                        self.known_resources.pop(pos, None)
                    return Action("collect")

        p_type = self.policy.get("type")
        if p_type == "random":
            return random_policy()
        if p_type == "greedy":
            if getattr(obs, "load", 0) == 0 and self.last_resource and self.last_resource in self.known_resources:
                return greedy_policy(self, obs, forced_target=self.last_resource)
            if getattr(obs, "load", 0) > 0 and self.last_nest:
                return greedy_policy(self, obs, forced_target=self.last_nest)
            return greedy_policy(self, obs)

        return Action("stay")

    def communicate(self, message: str, from_agent: "Agent"):
        print(f"[{self.name}] received message from {from_agent.name}: {message}")
