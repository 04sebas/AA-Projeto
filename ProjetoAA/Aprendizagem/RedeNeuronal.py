import random
import numpy as np

def computar_saidas_camada(entradas, pesos, vies, funcao_ativacao):
    z = np.dot(entradas, pesos) + vies
    saidas = funcao_ativacao(z)
    return saidas

def criar_arquitetura_rede(tamanho_entrada, tamanho_saida, neuronios):
    rn = RedeNeuronal(tamanho_entrada, tamanho_saida, neuronios, ativacao_oculta=relu, ativacao_saida=funcao_saida)
    num_pesos = rn.calcular_numero_pesos()
    pesos = [random.uniform(-1, 1) for _ in range(num_pesos)]
    rn.carregar_pesos(pesos)
    return rn

def relu(x):
    return np.maximum(0, x)

def funcao_saida(x):
    return x

class RedeNeuronal:

    def __init__(self, tamanho_entrada, tamanho_saida, arquitetura_oculta, ativacao_oculta, ativacao_saida):
        self.pesos_saida = None
        self.vies_saida = None
        self.pesos_ocultos = None
        self.vies_ocultos = None
        self.tamanho_entrada = tamanho_entrada
        self.arquitetura_oculta = arquitetura_oculta
        self.ativacao_oculta = ativacao_oculta
        self.ativacao_saida = ativacao_saida
        self.tamanho_saida = tamanho_saida
        self.pesos = None

    def calcular_numero_pesos(self):
        num_pesos = 0
        tamanho_atual = self.tamanho_entrada
        for camada_n in self.arquitetura_oculta:
            num_pesos += (tamanho_atual + 1) * camada_n
            tamanho_atual = camada_n
        num_pesos += (tamanho_atual + 1) * self.tamanho_saida
        return num_pesos

    def carregar_pesos(self, pesos):
        w = np.array(pesos)
        self.pesos = w
        self.pesos_ocultos = []
        self.vies_ocultos = []

        inicio_w = 0
        tamanho_atual = self.tamanho_entrada
        for n in self.arquitetura_oculta:
            fim_w = inicio_w + (tamanho_atual + 1) * n
            if fim_w > w.size:
                raise ValueError(f"carregar_pesos: tamanho insuficiente do vetor de pesos (necessário {fim_w}, obtido {w.size})")
            self.vies_ocultos.append(w[inicio_w:inicio_w + n])
            self.pesos_ocultos.append(w[inicio_w + n:fim_w].reshape(tamanho_atual, n))
            inicio_w = fim_w
            tamanho_atual = n

        fim_w = inicio_w + (tamanho_atual + 1) * self.tamanho_saida
        if fim_w > w.size:
            raise ValueError(f"carregar_pesos: tamanho insuficiente do vetor de pesos (necessário {fim_w}, obtido {w.size})")
        self.vies_saida = w[inicio_w:inicio_w + self.tamanho_saida]
        self.pesos_saida = w[inicio_w + self.tamanho_saida:fim_w].reshape(tamanho_atual, self.tamanho_saida)

    def propagar(self, x):
        a = x
        for pesos, vies in zip(self.pesos_ocultos, self.vies_ocultos):
            a = computar_saidas_camada(a, pesos, vies, self.ativacao_oculta)
        saida = computar_saidas_camada(a, self.pesos_saida, self.vies_saida, self.ativacao_saida)
        return saida

    def propagar_com_cache(self, x):
        ativacoes = [x]
        zs = []

        a = x
        for W, b in zip(self.pesos_ocultos, self.vies_ocultos):
            z = np.dot(a, W) + b
            zs.append(z)
            a = self.ativacao_oculta(z)
            ativacoes.append(a)

        z = np.dot(a, self.pesos_saida) + self.vies_saida
        zs.append(z)
        a = self.ativacao_saida(z)
        ativacoes.append(a)

        return a, ativacoes, zs

    def calcular_gradientes(self, x, alvo):
        output, ativacoes, zs = self.propagar_com_cache(x)

        def derivada_oculta(z):
            return np.where(z > 0, 1.0, 0.01)

        def derivada_saida():
            return 1

        delta = (output - alvo) * derivada_saida()

        grad_saida_W = np.outer(ativacoes[-2], delta)
        grad_saida_b = delta

        grad_oculto_W = []
        grad_oculto_b = []

        delta_l = delta

        # Iterar de trás para frente nas camadas ocultas
        for camada in reversed(range(len(self.pesos_ocultos))):
            z = zs[camada]
            a_prev = ativacoes[camada]

            d_act = derivada_oculta(z)

            if camada == len(self.pesos_ocultos) - 1:
                W_prox = self.pesos_saida
            else:
                W_prox = self.pesos_ocultos[camada + 1]

            delta_l = np.dot(delta_l, W_prox.T) * d_act

            grad_W_l = np.outer(a_prev, delta_l)
            grad_b_l = delta_l

            grad_oculto_W.insert(0, grad_W_l)
            grad_oculto_b.insert(0, grad_b_l)

        grads = []
        for Wg, bg in zip(grad_oculto_W, grad_oculto_b):
            grads.extend(bg.flatten())
            grads.extend(Wg.flatten())
        grads.extend(grad_saida_b.flatten())
        grads.extend(grad_saida_W.flatten())

        return np.array(grads)

    def achatar_params(self):
        return np.concatenate([w.flatten() for w in self.pesos])

    def desachatar_params(self, params_achatados):
        novos_pesos = []
        idx = 0
        for w in self.pesos:
            tamanho = w.size
            novo_w = params_achatados[idx:idx + tamanho].reshape(w.shape)
            novos_pesos.append(novo_w)
            idx += tamanho
        self.pesos = novos_pesos

    def achatar_grads(self, grads):
        return np.concatenate([g.flatten() for g in grads])


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.params = params
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.passos = 0

    def step(self, grads):
            self.passos += 1

            self.m = self.beta1 * self.m + (1 - self.beta1) * grads

            self.v = self.beta2 * self.v + (1 - self.beta2) * (grads * grads)

            m_hat = self.m / (1 - self.beta1 ** self.passos)
            v_hat = self.v / (1 - self.beta2 ** self.passos)

            self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
