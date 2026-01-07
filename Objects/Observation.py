class Observation:
    def __init__(self, perceptions=None):
        self.perceptions = perceptions
        self.position = (0, 0)
        self.width = 0
        self.height = 0

    def __repr__(self):
        if not self.perceptions:
            return "Nothing visible"
        return f"Observation({self.perceptions})"

    def contains(self, object_type: str) -> bool:
        return any(p["type"] == object_type for p in self.perceptions)
