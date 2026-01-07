from abc import ABC, abstractmethod
from Objects.Sensor import Sensor
from Objects.Action import Action

class Agent(ABC):
    def __init__(self, pos, name, policy=None):
        self.pos = pos
        self.name = name
        self.policy = policy or {}
        
        self.total_reward = 0.0
        self.resources_collected = 0
        self.resources_deposited = 0
        self.found_target = False
        self.trainable = False
        
        self.last_obs = None
        
        sensing_range = self.policy.get("range", 1)
        self.sensors = Sensor(sensing_range=sensing_range)

    @classmethod
    @abstractmethod
    def create(cls, config_filename: str):
        pass

    @abstractmethod
    def observation(self, obs):
        self.last_obs = obs

    @abstractmethod
    def act(self) -> Action:
        pass

    def evaluate_current_state(self, reward: float):
        self.total_reward += reward

    def install(self, sensor: Sensor):
        # Assuming list logic if meant to support multiple sensors, but code showed single assignment or append?
        # Original code: self.sensores.append(sensor) but self.sensores initialized as Sensor() object (not list)
        # Assuming logic was faulty or single sensor intended. I will stick to single sensor assignment or list if needed.
        # Looking at original: self.sensores = Sensor(...). self.instala calls append. This would crash in Python if Sensor is not list.
        # I'll fix it to assume single sensor for now or make it a list if that was the intent.
        # Given the usage in other files: 'agente.sensores.alcance' implies a single object.
        # So 'instala' was likely broken. I'll just assign it.
        self.sensors = sensor

    @abstractmethod
    def communicate(self, message: str, from_agent):
        pass
