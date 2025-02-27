# Module: custom_classes

# 原子类
class Atom:
    def __init__(self, name='atom', index=0, x=0, y=0, z=0, energy=0):
        self.name = name
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.energy = energy