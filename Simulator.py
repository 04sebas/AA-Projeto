import GeneticRun
from DQNRun import DqnSimulation


class Simulator:

    def __init__(self, input):
        self.agents = []
        self.farol = False
        if input == "Farol":
            self.farol = True
        
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

    def runSimulationFarolDqn(self):
        pass

if __name__ == "__main__":
    sim = Simulator("Farol")
    if sim.farol:
        print("RUNNING FAROL")
        sim.runSimulationFarol()
    else:
        print("No simulation selected.")
            
    
    

