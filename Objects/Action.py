class Action:
    def __init__(self, name, parameters=None):
        self.name = name or None
        self.parameters = parameters or {}

    def __repr__(self):
        return f"Action({self.parameters})"
