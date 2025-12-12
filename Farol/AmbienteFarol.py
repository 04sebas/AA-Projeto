import os
import random

from matplotlib import pyplot as plt, patches
import numpy as np

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

class Goal:
    def __init__(self, xx, yy):
        self.name = "Farol"
        self.x = xx
        self.y = yy

### Classe que representa o ambiente Farol ###

class Farol:

    ## Construtor da classe Farol --- lê o ficheiro de texto e cria o ambiente ( Tamanho fixo 100x100 ) ##
    def __init__(self, file):

        self.grid, self.goalx, self.goaly, self.walls = self.load_grid_from_file(file)
        self.size = 100
        self.agentx, self.agenty = self.random_valid_position()


    ## Função que carrega o grid a partir de um ficheiro de texto ##

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
        goalx = goaly = None
        walls = []

        for y, line in enumerate(lines): ## y inicial = 100
            mapped_y = height - 1 - y ## y mapeado = 0 na primeira iteração
            for x, ch in enumerate(line): ## Para cada caracter na linha prenche na grid, caso W -> Wall, F -> Goal, o resto é Ground
                if ch == 'W':
                    grid[mapped_y, x] = 1
                    walls.append(Wall(x, mapped_y))
                elif ch == 'F':
                    grid[mapped_y, x] = 2
                    goalx, goaly = x, mapped_y

                # O resto da grid fica a zero

        return grid, goalx, goaly, walls ## Retorna a grid, coordenadas do farol e lista de walls ##

    def get_object_here(self, x, y): ## Função que retorna o objeto na posição (x,y)
        objectInGrid = self.grid[y, x]
        if objectInGrid == 0:
            return Ground(x, y)
        elif objectInGrid == 1:
            return Wall(x, y)
        elif objectInGrid == 2:
            return Goal(x, y)
        return None

    def randomValidAction(self):
        actions = []
        #up
        if self.agenty + 1 < self.size:
            actions.append(0)
        #down
        if self.agenty - 1 > - 1:
            actions.append(1)
        #right
        if self.agentx + 1 < self.size:
            actions.append(2)
        #left
        if self.agentx - 1 > - 1:
            actions.append(3)

        return random.choice(actions)

    def random_valid_position(self):

        attempts = 0
        max_attempts = 1000  # safety limit

        while attempts < max_attempts:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            obj = self.get_object_here(x, y)
            if not isinstance(obj, (Wall, Goal)):
                return x, y
            attempts += 1

        return 0,0

    def reset(self):
        self.grid, self.goalx, self.goaly, self.walls = self.load_grid_from_file("Farol.txt")
        self.size = 100
        self.agentx, self.agenty = self.random_valid_position()

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


if __name__ == "__main__":
    farol = Farol("Farol.txt")
    ##### PATH PLOT #####
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot walls and goal
    for wall in farol.walls:
        ax.add_patch(
            patches.Rectangle(
                (wall.x - 0.5, wall.y - 0.5),  # lower-left corner
                1, 1,  # width, height
                facecolor='black'
            )
        )

    ax.text(farol.goalx, farol.goaly, "G", color='green', fontsize=9, ha='center', va='center', fontweight='bold')
    # Plot paths

    ax.set_xlim(-1, 100)
    ax.set_ylim(-1, 100)
    ax.set_title("Best Agent Paths Over Generations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid(True)
    plt.show(block=True)
