from Environments.Environment import Environment
from Objects.Observation import Observation
from Learning.Policies import DIRECTIONS

class FarolEnvironment(Environment):
    def __init__(self, width=100, height=100, farol_pos=(50,75), obstacles=None):
        super().__init__(width, height, obstacles=obstacles, name="FarolEnvironment")
        self.farol_pos = (int(farol_pos[0]), int(farol_pos[1]))

        if self.farol_pos in self.obstacles:
            self.obstacles.discard(self.farol_pos)

        self.resources = {self.farol_pos: {"type": "beacon", "pos": list(self.farol_pos), "value": 1500, "quantity": 1}}
        self.raw_obstacles = [{"pos": [p[0], p[1]]} for p in self.obstacles]
        self.targets = {}

    def observation_for(self, agent):
        pos = tuple(self.positions.get(agent, (0, 0)))
        perceptions = agent.sensors.perceive(self, pos) or []

        if pos == self.farol_pos:
            perceptions.append({
                "type": "beacon",
                "pos": pos
            })

        obs = Observation(perceptions)
        obs.position = pos
        obs.width = self.width
        obs.height = self.height
        agent.last_obs = obs
        obs.load = 0
        obs.goal = self.farol_pos
        obs.foraging = False
        return obs

    def act(self, action, agent):
        if getattr(agent, "found_target", False):
            return 0.0
        pos_raw = self.positions.get(agent, (0, 0))
        x, y = int(pos_raw[0]), int(pos_raw[1])

        if (x, y) == self.farol_pos:
            resource = self.resources.get((x, y))
            if resource:
                agent.found_target = True
                return float(resource.get("value", 1500))

        if action.name == "collect":
            return -0.5

        dx, dy = DIRECTIONS.get(action.name, (0, 0))
        newx, newy = x + int(dx), y + int(dy)

        if not self.is_valid_position(newx, newy):
            return -1

        old_dist = self.distance_to_target(x, y)
        self.positions[agent] = (newx, newy)
        agent.pos = (newx, newy)
        new_dist = self.distance_to_target(newx, newy)

        reward = -0.1
        if (newx, newy) == (x, y):
            reward -= 0.2

        if new_dist is not None and old_dist is not None and new_dist < old_dist:
            reward += 1.0

        return reward

    def distance_to_target(self, x, y):
        dist = abs(int(x) - self.farol_pos[0]) + abs(int(y) - self.farol_pos[1])
        max_dist = float(self.width + self.height) or 1.0
        return dist / max_dist

    def finished(self, agents=None):
        return all(getattr(a, "found_target", False) for a in agents)
