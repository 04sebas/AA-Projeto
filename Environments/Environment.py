from abc import ABC, abstractmethod
import random
from Agents.Agent import Agent
from Objects.Action import Action
from Objects.Observation import Observation


class Environment(ABC):
    def __init__(self, width, height, resources=None, obstacles=None, name="Environment"):
        self.width = int(width)
        self.height = int(height)
        self.name = name
        self.time = 0
        self.positions = {}
        
        self.resources = resources or {}
        
        self.obstacles = set()
        if obstacles:
            self._interpret_obstacles(obstacles)

    def _interpret_obstacles(self, obstacles):
        for o in obstacles:
            if isinstance(o, dict) and "pos" in o:
                self.obstacles.add(tuple(o["pos"]))
            elif isinstance(o, (list, tuple)) and len(o) >= 2:
                self.obstacles.add((int(o[0]), int(o[1])))

    @abstractmethod
    def observation_for(self, agent: Agent) -> Observation:
        pass

    @abstractmethod
    def act(self, action: Action, agent: Agent):
        pass

    def update(self):
        self.time += 1

    def restart(self):
        self.positions = {}
        self.time = 0

    @abstractmethod
    def finished(self, agents=None):
        pass

    def is_valid_position(self, x, y):
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        if (x, y) in self.obstacles:
            return False
        return True

    def random_position(self):
        step_x = self.width // 4
        step_y = self.height // 4
        positions = []

        for r in range(1, 4):
            for c in range(1, 4):
                pos = (step_x * c, step_y * r)
                if pos in self.obstacles or pos in self.resources:
                    continue
                positions.append(pos)

        if not positions:
            attempt = 0
            while True:
                pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
                if pos not in self.obstacles and pos not in self.resources:
                    return pos
                attempt += 1
                if attempt > 1000:
                    raise RuntimeError("No free positions.")
        return random.choice(positions)
