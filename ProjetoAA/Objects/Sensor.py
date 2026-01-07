class Sensor:
    def __init__(self, sensing_range: int = 3):
        self.sensing_range = sensing_range

    def perceive(self, environment, agent_position):
        visible_objects = []
        x, y = agent_position

        for dx in range(-self.sensing_range, self.sensing_range + 1):
            for dy in range(-self.sensing_range, self.sensing_range + 1):
                pos = (x + dx, y + dy)

                if not environment.is_valid_position(*pos):
                    continue

                if pos == (x, y):
                    continue

                if pos in environment.obstacles:
                    visible_objects.append({"type": "obstacle", "pos": pos})
                    continue

                if pos in environment.resources:
                    info = environment.resources[pos]
                    quantity = info.get("quantity", 0)
                    if quantity > 0:
                        visible_objects.append({
                            "type": "resource",
                            "pos": pos,
                            "value": info.get("value", 1),
                            "quantity": quantity
                        })
                    continue

                if hasattr(environment, "nests") and pos in environment.nests:
                    visible_objects.append({"type": "nest", "pos": pos})

                if hasattr(environment, "positions"):
                    from ProjetoAA.Agents.LearningAgent import LearningAgent
                    for agent, agent_pos in environment.positions.items():
                        if (
                            agent_pos == pos
                            and agent_pos != agent_position
                            and isinstance(agent, LearningAgent)
                        ):
                            visible_objects.append({
                                "type": "agent",
                                "pos": pos,
                                "ref": agent
                            })
                            break
                    continue

        return visible_objects
