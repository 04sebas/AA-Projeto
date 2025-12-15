import os
import random
from abc import ABC, abstractmethod

import numpy as np

from Projeto.Objetos.ObjetosAmbiente import Wall, Ground, Resource, DeliveryPoint, Goal


class Ambiente(ABC):

    def __init__(self, file):
        self.file = file

        self.goaly = None
        self.grid = None
        self.goalx = None
        self.size = None
        self.type = None
        self.resources = []
        self.deliveryPoints = []
        self.walls = []
        self.agentes = []

        self.load_grid_from_file(file)
        self.agentx, self.agenty = self.random_valid_position()

        ## Função que carrega o grid a partir de um ficheiro de texto ##

    @abstractmethod
    def observacaoPara(self, agente, depth = 3):  # Observação para agente, depende do ambiente
        pass

    @abstractmethod
    def atulalizacao(self):
        pass

    @abstractmethod
    def agir(self, acao, agente):  # Função que lida com a ação escolhida pelo agente, depende do ambiente
        pass

    def load_grid_from_file(self, filename):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, filename)

        ## Lê o ficheiro e cria a matriz que representa o ambiente ##
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        height = len(lines)
        width = len(lines[0])
        self.size = height

        ## Cria a grid inicial vazia do ambiente ##
        grid = np.zeros((height, width), dtype=np.int8)

        # --- Preenche o ambiente com os objetos correspondentes --- ##

        goalx = goaly = None
        walls = []

        for y, line in enumerate(lines):  ## y inicial = 100
            mapped_y = height - 1 - y  ## y mapeado = 0 na primeira iteração
            for x, ch in enumerate(
                    line):  ## Para cada caracter na linha prenche na grid, caso W -> Wall, F -> Goal, o resto é Ground
                if ch == 'W':
                    grid[mapped_y, x] = 1
                    self.walls.append(Wall(x, mapped_y))
                elif ch == 'R':
                    grid[mapped_y, x] = 2
                    self.resources.append(Resource(x, mapped_y))
                elif ch == 'P':
                    grid[mapped_y, x] = 3
                    self.deliveryPoints.append(DeliveryPoint(x, mapped_y))
                elif ch == 'F':
                    grid[mapped_y, x] = 4
                    goalx, goaly = x, mapped_y
                # O resto da grid fica a zero
        self.grid = grid
        self.goalx = goalx
        self.goaly = goaly

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
            return DeliveryPoint(x, y)
        elif objectInGrid == 4:
            return Goal(x, y)
        return None
    @abstractmethod
    def random_valid_position(self):
        pass

    def reset(self):
        self.load_grid_from_file(self.file)
        self.agentx, self.agenty = self.random_valid_position()
        self.agentes = []