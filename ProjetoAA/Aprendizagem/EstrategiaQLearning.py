import math
import random
from collections import deque
import numpy as np
from ProjetoAA.Agentes.AgenteAprendizagem import AgenteAprendizagem
from ProjetoAA.Aprendizagem.RedeNeuronal import criar_arquitetura_rede, Adam
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Aprendizagem.EstrategiaAprendizagem import EstrategiaAprendizagem


class EstrategiaQLearning(EstrategiaAprendizagem):
    def __init__(
        self,
        arq_rn=(15, 4, (16, 8)),
        episodios=100,
        passos_por_ep=500,
        gama=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        decaimento_epsilon=0.985,
        tamanho_lote=32,
        freq_atualizacao_alvo=50,
        tamanho_memoria=50000,
        taxa_aprendizagem=0.001,
    ):
        super().__init__(arq_rn=arq_rn, detalhado=True)
        self.episodios = int(episodios)
        self.passos_por_ep = int(passos_por_ep)
        self.gama = float(gama)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.decaimento_epsilon = float(decaimento_epsilon)
        self.tamanho_lote = int(tamanho_lote)
        self.freq_atualizacao_alvo = int(freq_atualizacao_alvo)
        self.memoria = deque(maxlen=tamanho_memoria)
        self.taxa_aprendizagem = float(taxa_aprendizagem)
        self.passos_otimizacao = 0
        self.politica = None
        self.alvo = None
        self.otimizador = None
        
        self.historico_epsilon = []

    def _construir_redes(self, tamanho_entrada, tamanho_saida, ocultas):
        self.politica = RedeDQN(tamanho_entrada, tamanho_saida, ocultas)
        self.alvo = RedeDQN(tamanho_entrada, tamanho_saida, ocultas)
        self.alvo.carregar_pesos(self.politica.obter_pesos().copy())
        self.otimizador = Adam(self.politica.rn.pesos, lr=self.taxa_aprendizagem)

    def escolher_acao(self, estado, acoes_possiveis):
       pass 

    def selecionar_acao(self, estado):
        estado = np.asarray(estado, dtype=np.float32)
        esperado = getattr(self.politica.rn, "tamanho_entrada", None)
        if esperado is None:
            # Fallback for some reason
            pass
        if estado.shape[0] != esperado:
            raise ValueError(f"[DQN] tamanho estado {estado.shape[0]} != entrada rede {esperado}")
        if random.random() < self.epsilon:
            return random.randint(0, self.politica.rn.tamanho_saida - 1)
        q = self.politica.propagar(estado)
        return int(np.argmax(q))

    def otimizar_modelo(self):
        if len(self.memoria) < self.tamanho_lote:
            return

        lote = random.sample(self.memoria, self.tamanho_lote)
        estados, acoes, recompensas, prox_estados, terminados = zip(*lote)
        estados = np.array(estados)
        prox_estados = np.array(prox_estados)
        recompensas = np.array(recompensas)
        terminados = np.array(terminados, dtype=np.float32)
        acoes = np.array(acoes, dtype=int)
        
        q_atual = np.zeros(self.tamanho_lote, dtype=np.float32)

        for i in range(self.tamanho_lote):
            qv = self.politica.propagar(estados[i])
            q_atual[i] = qv[acoes[i]]

        q_alvo = np.zeros(self.tamanho_lote, dtype=np.float32)

        for i in range(self.tamanho_lote):
            if terminados[i]:
                q_alvo[i] = recompensas[i]
            else:
                prox_q = self.alvo.propagar(prox_estados[i])
                q_alvo[i] = recompensas[i] + self.gama * float(np.max(prox_q))

        gradientes = [np.zeros_like(w) for w in self.politica.rn.pesos]

        for i in range(self.tamanho_lote):
            estado = estados[i]
            acao = acoes[i]
            if estado.shape[0] != self.politica.rn.tamanho_entrada:
                raise RuntimeError(f"nn_input mismatch: {estado.shape[0]} vs {self.politica.rn.tamanho_entrada}")
            valores_q = self.politica.propagar(estado)
            target = valores_q.copy()
            target[acao] = q_alvo[i]
            grad_i = self.politica.calcular_gradientes(estado, target)
            gradientes = [g + gi for g, gi in zip(gradientes, grad_i)]


        gradientes = [g / self.tamanho_lote for g in gradientes]
        grads_achatados = self.politica.rn.flatten_grads(gradientes)
        self.otimizador.step(grads_achatados)
        self.politica.rn.unflatten_params(self.otimizador.params)
        self.passos_otimizacao += 1

        if self.passos_otimizacao % self.freq_atualizacao_alvo == 0:
            self.alvo.carregar_pesos(self.politica.obter_pesos().copy())

    def treinar(self, ambiente, detalhado=True, alcance=None):
        self.detalhado = detalhado
        if alcance is not None:
            agente_amostra = AgenteAprendizagem(politica={"alcance": int(alcance)})
        else:
            default_alc = getattr(ambiente, "alcance_padrao_sensor", None)
            if default_alc is not None:
                agente_amostra = AgenteAprendizagem(politica={"alcance": int(default_alc)})
            else:
                agente_amostra = AgenteAprendizagem()
        tamanho_entrada = agente_amostra.obter_tamanho_entrada()
        tamanho_saida = len(agente_amostra.nomes_accao)
        ocultas = self.arq_rn[2] if len(self.arq_rn) >= 3 else (16, 8)
        self._construir_redes(tamanho_entrada, tamanho_saida, ocultas)

        if self.politica.rn.tamanho_entrada != tamanho_entrada:
            raise RuntimeError(
                f"[DQN] inconsistencia: entrada politica {self.politica.rn.tamanho_entrada} != esperado {tamanho_entrada}")

        self.historico_fitness = [] # rewards history
        self.historico_caminhos = []
        self.historico_epsilon = [] 

        inicio = [0,0]
        for ep in range(self.episodios):
            if hasattr(ambiente, "reiniciar"):
                ambiente.reiniciar()
            elif hasattr(ambiente, "reset"):
                ambiente.reset()

            agente = AgenteAprendizagem(politica={"alcance": agente_amostra.sensores.alcance})
            agente.pos = inicio
            # Fix: random start if needed, but keeping logic consistent
            ambiente.posicoes[agente] = tuple(agente.pos)
            agente.encontrou_objetivo = False
            obs = ambiente.observacao_para(agente)
            agente.observacao(obs)
            estado = agente.construir_entrada_rn(obs)
            recompensa_ep = 0.0
            caminho = [tuple(agente.pos)]

            for passo in range(self.passos_por_ep):
                idx_acao = self.selecionar_acao(estado)
                nome_acao = agente.nomes_accao[idx_acao]
                acc = Accao(nome_acao)

                try:
                    recompensa = ambiente.agir(acc, agente)
                    if recompensa is None:
                        recompensa = 0.0
                except Exception:
                    recompensa = -1.0

                obs_prox = ambiente.observacao_para(agente)
                agente.observacao(obs_prox)
                prox_estado = agente.construir_entrada_rn(obs_prox)
                estado = estado.astype(np.float32)
                prox_estado = prox_estado.astype(np.float32)
                terminou_flag = ambiente.terminou([agente])
                
                self.memoria.append((estado, idx_acao, float(recompensa), prox_estado, terminou_flag))
                # Warmup
                aquecimento = max(self.tamanho_lote * 2, 1000)
                if len(self.memoria) >= aquecimento:
                    self.otimizar_modelo()

                estado = prox_estado
                recompensa_ep += float(recompensa)
                caminho.append(tuple(agente.pos))

                if terminou_flag:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.decaimento_epsilon)
            
            self.historico_fitness.append(recompensa_ep)
            self.historico_caminhos.append(caminho)
            self.historico_epsilon.append(self.epsilon)

            if self.detalhado:
                print(f"[DQN] Ep {ep + 1}/{self.episodios} recompensa={recompensa_ep:.2f} eps={self.epsilon:.3f}")
        
        self.melhores_pesos = self.politica.obter_pesos().copy()
        self.melhor_rn = self.politica.rn

        # Usar gerar_graficos com callback para plots extra
        self.gerar_graficos(ambiente, titulo_fitness="Recompensa por Episódio", titulo_caminhos="Trajetórias — DQN", outros_plots=self._plots_extras)

        return self.melhores_pesos, self.melhor_rn

    def _plots_extras(self, plt):
        # Plotar Media Movel
        recompensas = self.historico_fitness
        if len(recompensas) > 5:
            plt.figure(figsize=(10, 4.5))
            plt.plot(recompensas, alpha=0.3, label="Bruto")
            janela = max(1, min(20, len(recompensas) // 10 or 1))
            if janela > 1:
                media_mov = np.convolve(np.array(recompensas), np.ones(janela) / janela, mode='valid')
                plt.plot(range(janela - 1, janela - 1 + len(media_mov)), media_mov, linewidth=2, label=f'Média móvel (j={janela})')
            plt.title("Média Móvel de Recompensas")
            plt.grid()
            plt.legend()
            plt.tight_layout()

        # Plotar Epsilon
        if self.historico_epsilon:
            plt.figure(figsize=(8, 3))
            plt.plot(self.historico_epsilon, marker='.', label='Epsilon')
            plt.title("Epsilon ao longo do treino")
            plt.xlabel("Episódio")
            plt.ylabel("Epsilon")
            plt.grid(True)
            plt.tight_layout()


class RedeDQN:
    def __init__(self, entrada, saida, ocultas):
        self.rn = criar_arquitetura_rede(entrada, saida, ocultas)
    def propagar(self, x):
        return self.rn.propagar(x)
    def calcular_gradientes(self, x, alvo):
        return self.rn.calcular_gradientes(x, alvo)
    def carregar_pesos(self, w):
        self.rn.carregar_pesos(w)
    def obter_pesos(self):
        return self.rn.pesos.copy()