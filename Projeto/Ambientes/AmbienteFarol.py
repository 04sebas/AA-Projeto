import random

from matplotlib import pyplot as plt, patches
from Projeto.Ambientes.Ambiente import Ambiente
from Projeto.Objetos.ObjetosAmbiente import Goal, Wall, Resource, DeliveryPoint


### Classe que representa o ambiente Farol ###

class AmbienteFarol(Ambiente):

    ## Construtor do Ambiente Farol --- lê o ficheiro de texto e cria o ambiente ( Tamanho fixo 100x100 ) ##
    def __init__(self, file):
        super().__init__(file)
        self.type = "Farol"

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

    def atulalizacao(self):
        pass

    def agir(self, acao, agente):  # Função que lida com a ação escolhida pelo agente, depende do ambiente
        action_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = action_map[acao]
        newx, newy = agente.x + dx, agente.y + dy

        if not (0 <= newx < self.size and 0 <= newy < self.size):
            return -1, False

        obj = self.get_object_here(newx, newy)

        if isinstance(obj, Goal):
            agente.foundGoal = True
            agente.y = newy
            agente.x = newx
            return 1500, True

        elif isinstance(obj, Wall):

            return -1, False
        else:
            prev_dist = agente.distanceToGoal(agente.x, agente.y)
            agente.y = newy
            agente.x = newx

            new_dist = agente.distanceToGoal(agente.x, agente.y)
            reward = -0.1
            if (newx, newy) == (agente.x, agente.y):
                reward -= 0.2

            if new_dist < prev_dist:
                reward += 1.0

            return reward, False

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
    farol = AmbienteFarol("../AmbientesFicheiros/Farol.txt")
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
