import math
import random
import numpy as np
from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Objetos.Accao import Accao

class AgenteAprendizagem(Agente):
    def __init__(self, nome="AG", politica=None, posicao=None, nomes_accao=None):
        posicao = posicao if posicao is not None else [0, 0]
        politica = politica or {}
        if "alcance" not in politica:
            politica["alcance"] = 1
            
        super().__init__(posicao, nome, politica)
        
        self.rede_neuronal = None
        self.pesos = None
        self.treinavel = True
        
        if nomes_accao is None:
            self.nomes_accao = ["cima", "baixo", "direita", "esquerda", "recolher", "depositar"]
        else:
            self.nomes_accao = list(nomes_accao)

        self.tipo_estrategia = self.politica.get("tipo_estrategia", "genetica")
        self.estrategia_conf = self.politica.get("estrategia_conf", {})
        
        self.memoria_recursos = set()
        self.memoria_ninhos = set()
        self.valor_ultimo_recurso = 0

    def observacao(self, obs):
        super().observacao(obs)
        pos_atual = tuple(getattr(obs, "posicao", None))
        percepcoes = getattr(obs, "percepcoes", []) or []

        for p in percepcoes:
            tipo = p.get("tipo")
            pos_t = tuple(p.get("pos"))

            if tipo in ("recurso", "farol"):
                quantidade = p.get("quantidade", 0)
                if quantidade and quantidade > 0:
                    self.memoria_recursos.add(pos_t)
                else:
                    self.memoria_recursos.discard(pos_t)

            elif tipo == "ninho":
                self.memoria_ninhos.add(pos_t)

            elif tipo == "agente":
                de_agente = p.get("ref")
                if de_agente is not None:
                    self.comunica("foraging", de_agente)

        if pos_atual in self.memoria_recursos:
            if not any(tuple(p.get("pos")) == pos_atual and p.get("tipo") in ("recurso", "farol") for p in percepcoes):
                self.memoria_recursos.discard(pos_atual)

    def cria(self, ficheiro_parametros):
        return self

    def age(self):
        if self.rede_neuronal is None or self.ultima_obs is None:
            return Accao(random.choice(self.nomes_accao[:4]))

        obs = self.ultima_obs
        px, py = obs.posicao
        percepcoes = obs.percepcoes or []

        if any(tuple(p.get("pos")) == (px, py) and p.get("tipo") in ("recurso", "farol") for p in percepcoes):
            if not getattr(obs, "carga", 0) > 0:
                return Accao("recolher")

        if any(tuple(p.get("pos")) == (px, py) and p.get("tipo") == "ninho" for p in percepcoes):
            if getattr(obs, "carga", 0) > 0:
                return Accao("depositar")

        if getattr(obs, "carga", 0) > 0:
            ninhos_visiveis = [tuple(p.get("pos")) for p in percepcoes if p.get("tipo") == "ninho"]
            if ninhos_visiveis:
                alvo = min(ninhos_visiveis, key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
                return Accao(self.__acao_mover_para(alvo, (px, py)))

        agentes_visiveis = [p.get("ref") for p in percepcoes if p.get("tipo") == "agente" and p.get("ref") is not None]
        if agentes_visiveis:
            for de_ag in agentes_visiveis:
                self.comunica("foraging", de_ag)

        if getattr(obs, "carga", 0) == 0:
            recursos_visiveis = [tuple(p.get("pos")) for p in percepcoes if p.get("tipo") in ("recurso", "farol")]
            if recursos_visiveis:
                alvo = min(recursos_visiveis, key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
                return Accao(self.__acao_mover_para(alvo, (px, py)))

        objetivo = getattr(obs, "goal", None)  # Note: Ambiente may still set 'goal' in observation
        if objetivo is None:
            return Accao(random.choice(self.nomes_accao[:4]))

        entrada_rn = self.construir_entrada_rn(self.ultima_obs)

        if self.tipo_estrategia in ("dqn", "genetica"):
            saida = self.rede_neuronal.propagar(entrada_rn)
            indice_acao = int(np.argmax(saida))
        else:
            return Accao("ficar")

        indice_acao = max(0, min(indice_acao, len(self.nomes_accao) - 1))
        return Accao(self.nomes_accao[indice_acao])

    def comunica(self, mensagem, de_agente):
        if mensagem == "farol":
            return

        if mensagem == "foraging":
            recursos_outro = getattr(de_agente, "memoria_recursos", None)
            entregas_outro = getattr(de_agente, "memoria_ninhos", None)

            if isinstance(recursos_outro, set):
                self.memoria_recursos.update(recursos_outro)

            if isinstance(entregas_outro, set):
                self.memoria_ninhos.update(entregas_outro)
            return

    def vizinhanca(self):
        obs = self.ultima_obs
        if not obs:
            tamanho_entrada = (2 * self.sensores.alcance + 1) ** 2 - 1
            return [-0.9] * tamanho_entrada

        px, py = obs.posicao
        alcance = self.sensores.alcance
        percepcoes = obs.percepcoes or []
        caracteristicas = []

        objetivo = getattr(obs, "goal", None)
        if objetivo is not None:
            gx, gy = objetivo
            dist_atual = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        else:
            gx = gy = None
            dist_atual = None

        for dy in range(-alcance, alcance + 1):
            for dx in range(-alcance, alcance + 1):
                if dx == 0 and dy == 0:
                    continue

                pos_verificar = (px + dx, py + dy)
                objeto = next((p for p in percepcoes if tuple(p["pos"]) == pos_verificar), None)

                if objeto:
                    tipo = objeto.get("tipo", "")
                    if tipo == "obstaculo":
                        caracteristicas.append(-0.9)
                    elif tipo in ("recurso", "farol"):
                        if getattr(obs, "carga", 0) <= 0:
                            caracteristicas.append(0.9)
                        else:
                            caracteristicas.append(0.1)
                    elif tipo == "ninho":
                        if getattr(obs, "carga", 0) > 0:
                            caracteristicas.append(1.0)
                        else:
                            caracteristicas.append(0.1)
                    else:
                        caracteristicas.append(0.0)
                else:
                    if gx is not None and gy is not None:
                        dist_celula = math.sqrt((pos_verificar[0] - gx) ** 2 + (pos_verificar[1] - gy) ** 2)
                        if dist_celula < dist_atual:
                            caracteristicas.append(0.9)
                        else:
                            caracteristicas.append(-0.9)
                    else:
                        caracteristicas.append(-0.9)

        return caracteristicas

    def construir_entrada_rn(self, obs):
        px, py = obs.posicao
        caracteristicas = np.array(self.vizinhanca(), dtype=np.float32)

        largura = max(1.0, getattr(obs, "largura", 1))
        altura = max(1.0, getattr(obs, "altura", 1))

        norm_x = px / largura
        norm_y = py / altura

        objetivo = getattr(obs, "goal", None)
        if objetivo is not None:
            gx, gy = objetivo
            obj_x = (gx - px) / largura
            obj_y = (gy - py) / altura
        else:
            obj_x = 0.0
            obj_y = 0.0

        carregando = float(getattr(obs, "carga", 0) > 0)

        return np.concatenate(([norm_x, norm_y], caracteristicas, [obj_x, obj_y, carregando])).astype(np.float32)

    def obter_tamanho_entrada(self):
        alcance = self.sensores.alcance
        num_caracteristicas = (2 * alcance + 1) ** 2 - 1
        return int(num_caracteristicas + 5)

    def set_action_space(self, nomes_accao):
        self.nomes_accao = list(nomes_accao)

    def __acao_mover_para(self, alvo, atual):
        tx, ty = alvo
        cx, cy = atual
        dx = tx - cx
        dy = ty - cy

        if abs(dx) > abs(dy):
            if dx > 0:
                return "direita"
            elif dx < 0:
                return "esquerda"
        else:
            if dy > 0:
                return "baixo"
            elif dy < 0:
                return "cima"
        return "ficar"
