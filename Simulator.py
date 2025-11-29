import GeneticRun

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

    def runSimulationFarolGenetic(self):
        print("DEBUG: Starting Genetic Algorithm")
        a = GeneticRun.GeneticSimulation()
        a.geneticRun(True)

    def runSimulationFarolDqn(self):
        pass

if __name__ == "__main__":
    sim = Simulator("Farol")
    if sim.farol:
        print("RUNNING FAROL")
        sim.runSimulationFarolGenetic()
    else:
        print("No simulation selected.")
            
    
    

