import random
import numpy as np
from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
from ProjetoAA.Ambientes.AmbienteFarol import AmbienteFarol
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Aprendizagem.RedeNeuronal import criar_arquitetura_rede
from ProjetoAA.Aprendizagem.EstrategiaAprendizagem import EstrategiaAprendizagem


class EstrategiaGenetica(EstrategiaAprendizagem):
    def __init__(
        self,
        tamanho_populacao=100,
        taxa_mutacao=0.01,
        num_geracoes=25,
        tamanho_torneio=3,
        elitismo_frac=0.2,
        arq_rn=(15, 4, (16, 8)),
        passos_por_avaliacao=1000,
        desvio_mutacao=0.1,
    ):
        super().__init__(arq_rn=arq_rn, detalhado=True)
        self.tamanho_populacao = int(tamanho_populacao)
        self.taxa_mutacao = float(taxa_mutacao)
        self.num_geracoes = int(num_geracoes)
        self.tamanho_torneio = int(tamanho_torneio)
        self.elitismo_frac = float(elitismo_frac)
        self.passos_por_avaliacao = int(passos_por_avaliacao)
        self.desvio_mutacao = float(desvio_mutacao)

        self.num_pesos = None
        self.populacao = None
        self.aptidao = None
        self.treinada = False

    def escolher_acao(self, estado, acoes_possiveis):
        pass

    def _selecionar_torneio(self):
        competidores = random.sample(range(self.tamanho_populacao), self.tamanho_torneio)
        melhor = max(competidores, key=lambda i: self.aptidao[i])
        return melhor

    def _cruzamento(self, p1, p2):
        if self.num_pesos <= 1:
            return p1.copy(), p2.copy()
        ponto = random.randint(1, self.num_pesos - 1)
        filho1 = np.concatenate([p1[:ponto], p2[ponto:]]).astype(np.float32)
        filho2 = np.concatenate([p2[:ponto], p1[ponto:]]).astype(np.float32)
        return filho1, filho2

    def _mutar_pesos(self, pesos):
        w = pesos.copy()
        mascara = np.random.rand(self.num_pesos) < self.taxa_mutacao
        if np.any(mascara):
            w[mascara] += np.random.randn(np.sum(mascara)).astype(np.float32) * self.desvio_mutacao
        return w

    def _inicializar_populacao(self):
        rn_proto = criar_arquitetura_rede(*self.arq_rn)
        self.num_pesos = int(rn_proto.calcular_numero_pesos())
        pop = []
        for _ in range(self.tamanho_populacao):
            w = np.random.uniform(-1.0, 1.0, size=(self.num_pesos,)).astype(np.float32)
            pop.append(w)
        self.populacao = pop
        self.aptidao = np.zeros(self.tamanho_populacao, dtype=np.float32)

    def treinar(self, ambiente, detalhado=True, tamanho_entrada=3):
        self.detalhado = detalhado
        agente_falso = AgenteAprendizagem(politica={"alcance": 3})
        if hasattr(ambiente, "obter_nomes_acoes"):
            agente_falso.set_action_space(ambiente.obter_nomes_acoes())
        num_acoes = len(agente_falso.nomes_accao)
        acoes_disponiveis = agente_falso.nomes_accao
        alcance_sensor = getattr(agente_falso.sensores, "alcance", 3)
        self.arq_rn = (tamanho_entrada, num_acoes, self.arq_rn[2] if len(self.arq_rn) > 2 else (16, 8))

        self._inicializar_populacao()

        self.historico_fitness = []
        self.historico_caminhos = []
        
        n_elite = max(1, int(self.elitismo_frac * self.tamanho_populacao))

        if self.detalhado:
            print(f"[AG] População: {self.tamanho_populacao}, Gerações: {self.num_geracoes}, Elite: {n_elite}")

        for gen in range(self.num_geracoes):
            if self.detalhado:
                print(f"[AG] Geração {gen + 1}/{self.num_geracoes}")

            agentes_por_geracao = []

            for i, pesos in enumerate(self.populacao):
                ambiente.reiniciar()
                rn = criar_arquitetura_rede(*self.arq_rn)
                rn.carregar_pesos(pesos)

                agente = AgenteAprendizagem(politica={"alcance": alcance_sensor}, nomes_accao=acoes_disponiveis)
                agente.rede_neuronal = rn
                agente.pesos = pesos.copy()

                agente.pos = None # Reset pos will be handled
                ambiente.posicoes[agente] = None

                agente.encontrou_objetivo = False
                agente.recompensa_total = 0.0
                agente.recursos_recolhidos = 0
                agente.recursos_depositados = 0

                if isinstance(ambiente, AmbienteFarol):
                    inicio = ambiente.posicao_aleatoria()
                else:
                    inicio = (0, 0) # Default foraging

                ambiente.posicoes[agente] = tuple(inicio)
                agente.pos = tuple(inicio)
                caminho = [agente.pos]
                fitness = 0.0

                for passo in range(self.passos_por_avaliacao):
                    obs = ambiente.observacao_para(agente)
                    agente.observacao(obs)
                    acc = agente.age()

                    recompensa = ambiente.agir(acc, agente) or None
                    fitness += float(recompensa)
                    caminho.append(agente.pos)
                    terminou_flag = ambiente.terminou([agente])

                    if terminou_flag:
                        break
                
                # Adding manual attribute path to agent just for recording
                agente.caminho = caminho
                self.aptidao[i] = fitness
                agentes_por_geracao.append(agente)

            ordem = np.argsort(-self.aptidao)
            self.populacao = [self.populacao[idx] for idx in ordem]
            self.aptidao = self.aptidao[ordem]
            agentes_por_geracao = [agentes_por_geracao[idx] for idx in ordem]

            melhor_fit = float(self.aptidao[0])
            media_fit = float(np.mean(self.aptidao))
            
            # Atualiza historicos para gerar graficos
            self.historico_fitness.append(media_fit)
            if agentes_por_geracao:
                 melhor_agente = agentes_por_geracao[0]
                 self.historico_caminhos.append(getattr(melhor_agente, "caminho", []))

            if self.detalhado:
                print(f"[AG] Ger {gen + 1}: melhor={melhor_fit:.2f}, media={media_fit:.2f}")

            nova_pop = []
            for k in range(n_elite):
                nova_pop.append(self.populacao[k].copy())

            while len(nova_pop) < self.tamanho_populacao:
                pai1_idx = self._selecionar_torneio()
                pai2_idx = self._selecionar_torneio()
                p1 = self.populacao[pai1_idx].copy()
                p2 = self.populacao[pai2_idx].copy()

                filho1, filho2 = self._cruzamento(p1, p2)
                filho1 = self._mutar_pesos(filho1)
                if len(nova_pop) < self.tamanho_populacao:
                    nova_pop.append(filho1)
                if len(nova_pop) < self.tamanho_populacao:
                    filho2 = self._mutar_pesos(filho2)
                    nova_pop.append(filho2)

            self.populacao = nova_pop
            self.aptidao = np.zeros(self.tamanho_populacao, dtype=np.float32)

        melhores_pesos = self.populacao[0].copy()
        melhor_rn = criar_arquitetura_rede(*self.arq_rn)
        melhor_rn.carregar_pesos(melhores_pesos)

        self.melhores_pesos = melhores_pesos
        self.melhor_rn = melhor_rn
        self.treinada = True

        self.gerar_graficos(ambiente, titulo_fitness="Aptidão Média por Geração", titulo_caminhos="Evolução dos Melhores Caminhos")

        return melhores_pesos, melhor_rn
