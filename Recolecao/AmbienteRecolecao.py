import os
import random
from matplotlib import pyplot as plt, patches
import numpy as np
from Recolecao.AgentRecolecao import AgenteRecolecao
from Recolecao.Entities import Wall, Ground, Resource, Delivery

### Classe que representa o ambiente Farol ###

class AmbienteRecolecao:

    ## Construtor da classe Recolecao --- lê o ficheiro de texto e cria o ambiente ( Tamanho fixo 50x50 ) ##
    def __init__(self, file):

        self.grid, self.walls, self.resources, self.deliveryPoints = self.load_grid_from_file(file)
        self.size = 50
        self.agentes = []

    def load_grid_from_file(self, filename):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, filename)

        ## Lê o ficheiro e cria a matriz que representa o ambiente ##
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        height = len(lines)
        width = len(lines[0])

        ## Cria a grid inicial vazia do ambiente ##
        grid = np.zeros((height, width), dtype=np.int8)

        #--- Preenche a grid com os objetos correspondentes --- ##
        resources = []
        deliveryPoints = []
        walls = []

        for y, line in enumerate(lines): ## y inicial = 100
            mapped_y = height - 1 - y ## y mapeado = 0 na primeira iteração
            for x, ch in enumerate(line): ## Para cada caracter na linha prenche na grid, caso W -> Wall, F -> Goal, o resto é Ground
                if ch == 'W':
                    grid[mapped_y, x] = 1
                    walls.append(Wall(x, mapped_y))
                elif ch == 'R':
                    grid[mapped_y, x] = 2
                    resources.append(Resource(x, mapped_y))
                elif ch == 'P':
                    grid[mapped_y, x] = 3
                    deliveryPoints.append(Delivery(x, mapped_y))
                # O resto da grid fica a zero

        return grid , walls, resources, deliveryPoints ## Retorna a grid

    def get_object_here(self, x, y): ## Função que retorna o objeto na posição (x,y)

        for agente in self.agentes:
            if agente.x == x and agente.y == y:
                return agente

        objectInGrid = self.grid[y, x]
        if objectInGrid == 0:
            return Ground(x, y)
        elif objectInGrid == 1:
            return Wall(x, y)
        elif objectInGrid == 2:
            return Resource(x, y)
        elif objectInGrid == 3:
            return Delivery(x, y)
        return None

    def random_valid_position(self):

        attempts = 0
        max_attempts = 1000  # safety limit

        while attempts < max_attempts:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            obj = self.get_object_here(x, y)
            if isinstance(obj, Ground):
                return x, y
            attempts += 1

        return 0,0

    def reset(self):
        self.grid, self.walls, self.resources, self.deliveryPoints = self.load_grid_from_file("Recolecao.txt")
        self.size = 50

    def observation(self, x, y, depth=3):
        obs = {0: [], 1: [], 2: [], 3: []}  # up, down, left, right

        # direction vectors (dx, dy)
        directions = {
            0: (0, 1),  # up
            1: (0, -1),  # down
            2: (-1, 0),  # left
            3: (1, 0),  # right
        }

        for direction, (dx, dy) in directions.items():
            for step in range(1, depth + 1):
                nx = x + dx * step
                ny = y + dy * step

                if 0 <= nx < self.size and 0 <= ny < self.size:
                    obj = self.get_object_here(nx, ny)
                    obs[direction].append(obj)
                else:
                    obs[direction].append(None)

        return obs

    def consume(self, obj):
        self.grid[obj.y, obj.x] = 0

if __name__ == "__main__":
    rec = AmbienteRecolecao("Recolecao.txt")
    ##### PATH PLOT #####
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot walls and goal
    for wall in rec.walls:
        ax.add_patch(
            patches.Rectangle(
                (wall.x - 0.5, wall.y - 0.5),  # lower-left corner
                1, 1,  # width, height
                facecolor='black'
            )
        )

    for resource in rec.resources:
        ax.text(resource.x, resource.y, "R", color='green', fontsize=9, ha='center', va='center', fontweight='bold')

    for delivery in rec.deliveryPoints:
        ax.text(delivery.x, delivery.y, "P", color='red', fontsize=9, ha='center', va='center', fontweight='bold')

    for agente in rec.agentes:
        ax.text(agente.x, agente.y, "A", color='blue', fontsize=9, ha='center', va='center', fontweight='bold')
    # Plot paths

    ax.set_xlim(-1, 50)
    ax.set_ylim(-1, 50)
    ax.set_title("Recolecao")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid(True)
    plt.show(block=True)
