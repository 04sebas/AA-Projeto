
from Farol.DQNRun import DqnSimulation
from Recolecao import GeneticRecolecaoRun, DQNRecolecao


class Simulator:

    def __init__(self, input):
        self.agents = []
        self.farol = False
        if input == "Farol":
            self.farol = True

        if input == "Recolecao":
            self.recolecao = True
        
    def addAgent(self, a):
        self.agents.append(a)
        
    def listaAgentes(self):
        for agente in self.agents:
            print(agente)

    def runSimulationFarol(self):
        #print("DEBUG: Starting Genetic Algorithm")
        #a = GeneticRun.GeneticSimulation()
        #a.geneticRun(True)
        a = DqnSimulation()
        a.dqnRun()

    def runSimulationRecolecao(self):
        print("DEBUG: Starting Genetic Algorithm")
        #a = GeneticRecolecaoRun.GeneticSimulation()
        #a.geneticRun(True)
        a = DQNRecolecao.DqnSimulation()
        a.dqnRun()

if __name__ == "__main__":
    sim = Simulator("Recolecao")
    if sim.farol:
        print("RUNNING FAROL")
        sim.runSimulationFarol()
    if sim.recolecao:

        print("RUNNING RECOLECAO")
        sim.runSimulationRecolecao()

    else:
        print("No simulation selected.")
            
    
    

