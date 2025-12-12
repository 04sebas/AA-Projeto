### Classes que representam os objetos no ambiente ###

class Ground:
    def __init__(self, x, y):
        self.name = "Ground"
        self.x = x
        self.y = y

class Wall:
    def __init__(self, x, y):
        self.name = "Wall"
        self.x = x
        self.y = y

class Resource:
    def __init__(self, xx, yy):
        self.name = "Resource"
        self.x = xx
        self.y = yy

class Delivery:
    def __init__(self, xx, yy):
        self.name = "Delivery"
        self.x = xx
        self.y = yy
