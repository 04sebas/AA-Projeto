import math
from copy import deepcopy
from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Ambientes.Ambiente import Ambiente
from ProjetoAA.Aprendizagem.Politicas import DIRECOES
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Objetos.Observacao import Observacao

class AmbienteForaging(Ambiente):
    def __init__(self, largura=50, altura=50, recursos=None, ninhos=None, obstaculos=None):
        self.recursos_iniciais = {
            tuple(r["pos"]): {"valor": r["valor"], "quantidade": r["quantidade"]}
            for r in (recursos or [])
        }
        super().__init__(largura, altura, recursos=deepcopy(self.recursos_iniciais), obstaculos=obstaculos, nome="AmbienteForaging")
        self.capacidade_carga = 1
        self.ninhos = [tuple(n) for n in (ninhos or [])]
        self.cargas = {}
        self.alvos = {}

    def observacao_para(self, agente):
        pos = self.posicoes.get(agente, (0, 0))
        percepcoes = agente.sensores.perceber(self, pos)
        
        alvo = self.alvos.get(agente)
        if alvo and alvo not in self.recursos and alvo not in self.ninhos:
            self.alvos[agente] = None

        if pos in self.recursos:
            r = self.recursos[pos]
            qtd = r.get("quantidade", 0)
            if qtd > 0:
                percepcoes.append({"pos": pos, "tipo": "recurso", "valor": r.get("valor", 1), "quantidade": qtd})

        if pos in self.ninhos:
            percepcoes.append({"pos": pos, "tipo": "ninho"})

        alcance = getattr(agente.sensores, "alcance", 3)
        for outro, outra_pos in self.posicoes.items():
            if outro is agente: continue
            if abs(outra_pos[0] - pos[0]) <= alcance and abs(outra_pos[1] - pos[1]) <= alcance:
                percepcoes.append({"pos": tuple(outra_pos), "tipo": "agente", "ref": outro})

        if not self.alvos.get(agente):
            com_carga = self.cargas.get(agente, 0) > 0
            alvo = self._ninho_mais_proximo(pos) if com_carga else self._recurso_mais_proximo(pos)
            if alvo: self.alvos[agente] = alvo

        obs = Observacao(percepcoes)
        obs.posicao = pos
        obs.carga = self.cargas.get(agente, 0)
        obs.largura = self.largura
        obs.altura = self.altura
        obs.goal = self.alvos.get(agente)
        agente.ultima_obs = obs
        obs.foraging = True
        return obs

    def agir(self, accao: Accao, agente: Agente):
        pos = list(self.posicoes.get(agente, (0, 0)))
        dx, dy = DIRECOES.get(accao.nome, (0, 0))
        nova_pos = (pos[0] + dx, pos[1] + dy)

        if not self.posicao_valida(*nova_pos):
            return -1

        pos_atual = tuple(pos)
        alvo = self.alvos.get(agente)
        com_carga = self.cargas.get(agente, 0) > 0
        
        if not alvo:
            alvo = self._ninho_mais_proximo(pos_atual) if com_carga else self._recurso_mais_proximo(pos_atual)
            if alvo: self.alvos[agente] = alvo

        dist_antiga = self._distancia_normalizada(pos_atual, alvo)
        self.posicoes[agente] = nova_pos
        agente.pos = list(nova_pos)
        
        if accao.nome == "recolher" and nova_pos in self.recursos:
            carga = self.cargas.get(agente, 0)
            if carga < self.capacidade_carga:
                recurso = self.recursos[nova_pos]
                recompensa = 100.0 + recurso.get("valor", 25)
                agente.valor_ultimo_recurso = int(recurso.get("valor", 1))
                self.cargas[agente] = carga + 1
                
                if hasattr(agente, "recursos_recolhidos"): agente.recursos_recolhidos += 1
                
                recurso["quantidade"] = max(0, int(recurso.get("quantidade", 1)) - 1)
                if recurso["quantidade"] <= 0:
                    self._invalidar_alvos_para_recurso(nova_pos)

                self.alvos[agente] = self._ninho_mais_proximo(nova_pos) if self.cargas[agente] >= self.capacidade_carga else (self._recurso_mais_proximo(nova_pos) or self._ninho_mais_proximo(nova_pos))
                return recompensa
            else:
                return 1.0
        
        elif accao.nome == "depositar" and nova_pos in self.ninhos:
            carga = self.cargas.get(agente, 0)
            if carga > 0:
                recompensa = 200.0 + float(getattr(agente, "valor_ultimo_recurso", 1))
                self.cargas[agente] = 0
                agente.valor_ultimo_recurso = 0
                if hasattr(agente, "recursos_depositados"): agente.recursos_depositados += carga
                self.alvos[agente] = self._recurso_mais_proximo(nova_pos)
                return recompensa
            else:
                return 1.0

        elif accao.nome == "ficar":
            return -0.3

        alvo_depois = self.alvos.get(agente)
        nova_dist = self._distancia_normalizada(nova_pos, alvo_depois)
        
        recompensa = -0.1
        if dist_antiga is not None and nova_dist is not None:
             if nova_dist < dist_antiga: recompensa += 0.5
             elif nova_dist > dist_antiga: recompensa -= 0.05
        return recompensa

    def reiniciar(self):
        super().reiniciar()
        self.cargas = {}
        self.alvos = {}
        self.recursos = deepcopy(self.recursos_iniciais)

    def _recurso_mais_proximo(self, pos):
        if not self.recursos: return None
        melhor, melhor_d = None, None
        for rpos, info in self.recursos.items():
            if info.get("quantidade", 0) <= 0: continue
            d = math.hypot(rpos[0]-pos[0], rpos[1]-pos[1])
            if melhor_d is None or d < melhor_d:
                melhor_d = d
                melhor = rpos
        return melhor

    def _ninho_mais_proximo(self, pos):
        if not self.ninhos: return None
        melhor, melhor_d = None, None
        for npos in self.ninhos:
            d = math.hypot(npos[0]-pos[0], npos[1]-pos[1])
            if melhor_d is None or d < melhor_d:
                melhor_d = d
                melhor = npos
        return melhor

    def _distancia_normalizada(self, pos, alvo):
        if not alvo: return None
        dist = math.hypot(pos[0]-alvo[0], pos[1]-alvo[1])
        max_dist = math.hypot(self.largura, self.altura) or 1.0
        return dist / max_dist

    def _invalidar_alvos_para_recurso(self, pos_recurso):
        for ag, tgt in list(self.alvos.items()):
            if tgt == pos_recurso:
                pos_agente = self.posicoes.get(ag)
                self.alvos[ag] = self._ninho_mais_proximo(pos_agente) if (pos_agente and self.cargas.get(ag, 0) > 0) else self._recurso_mais_proximo(pos_agente)

    def terminou(self, agentes=None) -> bool:
        recursos_esgotados = all(info.get("quantidade", 0) <= 0 for info in self.recursos.values())
        agentes_a_verificar = agentes if agentes is not None else list(self.posicoes.keys())
        nenhum_agente_com_carga = all(self.cargas.get(ag, 0) == 0 for ag in agentes_a_verificar)
        return recursos_esgotados and nenhum_agente_com_carga

