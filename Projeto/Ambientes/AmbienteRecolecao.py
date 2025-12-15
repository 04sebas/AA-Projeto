import random

from matplotlib import pyplot as plt, patches
from Projeto.Ambientes.Ambiente import Ambiente
from Projeto.Objetos.ObjetosAmbiente import Resource, DeliveryPoint, Wall, Goal


### Classe que representa o ambiente Farol ###

class AmbienteRecolecao(Ambiente):

    ## Construtor do ambiente Recolecao --- lê o ficheiro de texto e cria o ambiente ( Tamanho fixo 50x50 ) ##
    def __init__(self, file):
        super().__init__(file)
        self.type = "Recolecao"
        self.goalStepsWithoutProgress = 0
        self.lastGoalDistance = None
        self.maxGoalSteps = 75

    def observacaoPara(self, agente, depth = 3):  # Observação para agente, depende do ambiente
        size = 2 * depth + 1
        obs = [[None for _ in range(size)] for _ in range(size)]

        for dx in range(-depth, depth + 1):
            for dy in range(-depth, depth + 1):
                nx = agente.x + dx
                ny = agente.y + dy

                obs[dy + depth][dx + depth] = (
                    self.get_object_here(nx, ny)
                    if 0 <= nx < self.size and 0 <= ny < self.size
                    else None
                )
        return obs

    def consume(self, obj):
        self.grid[obj.y, obj.x] = 0

    def atulalizacao(self):
        self.goalStepsWithoutProgress += 1

    def agir(self, acao, agente):  # Função que lida com a ação escolhida pelo agente, depende do ambiente

        action_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = action_map[acao]
        newx, newy = agente.x + dx, agente.y + dy

        if not (0 <= newx < self.size and 0 <= newy < self.size):
            return -1, False

        obj = self.get_object_here(newx, newy)
        agente.update_goal_from_observation()

        if isinstance(obj, Resource):
            agente.y = newy
            agente.x = newx
            if not agente.carrying: # Não leva nada - > carrega o resource
                self.consume(obj)
                agente.carrying = True
                agente.currentGoal = None
                self.goalStepsWithoutProgress = 0
                self.lastGoalDistance = None
                agente.resources.discard(obj)
                return 100, False
            else:
                return 1, False

        elif isinstance(obj, DeliveryPoint):
            agente.y = newy
            agente.x = newx
            if agente.carrying:
                agente.carrying = False
                agente.currentGoal = None
                self.goalStepsWithoutProgress = 0
                self.lastGoalDistance = None
                agente.delivered += 1
                return 200, False
            else:
                return 1, False
        elif isinstance(obj, Wall):
            return -1, False
        else:
            agente.y = newy
            agente.x = newx
            return 0.1, False

    def random_valid_position(self):
        attempts = 0
        max_attempts = 1000  # safety limit

        while attempts < max_attempts:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            obj = self.get_object_here(x, y)

            if not isinstance(obj, (Wall, Goal, Resource, DeliveryPoint)):
                return x, y
            attempts += 1

        return 0,0
if __name__ == "__main__":
    rec = AmbienteRecolecao("../AmbientesFicheiros/Recolecao.txt")
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
