from Projeto.Aprendizagem.EstrategiaDqn import EstrategiaDqn
from Projeto.Aprendizagem.EstrategiaGenetica import EstrategiaGenetica


class MotorDeSimulator:

    def __init__(self):
        self.agents = []
        self.ambiente = None
        self.modo = None
        self.estrategia = None

    def cria(self, string):
        pass

    def listaAgentes(self):
        for agente in self.agents:
            print(agente)

    def executa(self):
        pass

    def dqnForaging(self, input = 16):
        a = EstrategiaDqn(input)
        a.dqnRun("../AmbientesFicheiros/Recolecao.txt", "Recolecao")

    def dqnFarol(self, input = 15):
        a = EstrategiaDqn(input)
        a.dqnRun("../AmbientesFicheiros/Farol.txt", "Farol")

    def geneticFarol(self, input = 15):
        a = EstrategiaGenetica
        a.algoritmoGenetico(a, "../AmbientesFicheiros/Farol.txt", input, "Farol")

    def geneticForaging(self, input = 16):
        a = EstrategiaGenetica
        a.algoritmoGenetico("../AmbientesFicheiros/Recolecao.txt", input, "Recolecao")
if __name__ == "__main__":
    motor = MotorDeSimulator()
    motor.geneticFarol()
    

