import json
from ProjetoAA.Agentes.Agente import Agente
from ProjetoAA.Aprendizagem.Politicas import DIRECOES, politica_aleatoria, politica_greedy
from ProjetoAA.Objetos.Accao import Accao
from ProjetoAA.Objetos.Observacao import Observacao

class AgenteFixo(Agente):
    def __init__(self, posicao=None, politica=None, nome="AF"):
        posicao = posicao if posicao is not None else [0, 0]
        politica = politica or {}
        if "alcance" not in politica:
            politica["alcance"] = 5
            
        super().__init__(posicao, nome, politica)
        
        self.treinavel = False
        self.ultima_direcao = None
        self.ultimo_ninho = None
        self.posicao_prev = None
        self.passos_sem_mover = 0
        self.recursos_conhecidos = {}
        self.ultimo_recurso = None
        self.obstaculos_conhecidos = set()
        self.limite_imobilidade = self.politica.get("limite_imobilidade", 2)

    @classmethod
    def cria(cls, nome_do_ficheiro_parametros: str) -> "AgenteFixo":
        try:
            with open(nome_do_ficheiro_parametros, 'r', encoding='utf-8') as f:
                config = json.load(f)

            agentes_config = config.get("agentes", [])

            for conf in agentes_config:
                if conf.get("tipo") == "AgenteFixo":
                    politica_real = conf.get("politica", {})

                    return cls(
                        politica=politica_real
                    )

            return cls()
        except Exception as e:
            print(f"Erro a criar AgenteFixo a partir do ficheiro: {e}")
            return cls()

    def observacao(self, obs: Observacao):
        super().observacao(obs)

        pos_atual = tuple(getattr(obs, "posicao", None))
        if self.posicao_prev is None:
            self.passos_sem_mover = 0
        else:
            if pos_atual == self.posicao_prev:
                self.passos_sem_mover += 1
            else:
                self.passos_sem_mover = 0
        self.posicao_prev = pos_atual

        percepcoes = getattr(obs, "percepcoes", []) or []
        for p in percepcoes:
            tipo = p.get("tipo")
            if tipo in ("recurso", "farol"):
                pos_t = tuple(p.get("pos"))
                quantidade = p.get("quantidade", p.get("valor", 1))
                if quantidade and quantidade > 0:
                    self.recursos_conhecidos[pos_t] = quantidade
                else:
                    self.recursos_conhecidos.pop(pos_t, None)
            if tipo == "ninho":
                self.ultimo_ninho = tuple(p.get("pos"))

        if pos_atual in self.recursos_conhecidos:
            if not any(tuple(p.get("pos")) == pos_atual and p.get("tipo") in ("recurso", "farol")
                       for p in percepcoes):
                self.recursos_conhecidos.pop(pos_atual, None)
                if self.ultimo_recurso == pos_atual:
                    self.ultimo_recurso = None

        if self.passos_sem_mover >= self.limite_imobilidade and self.ultima_direcao:
            delta = DIRECOES.get(self.ultima_direcao, (0, 0))
            if pos_atual is not None:
                tentativa = (pos_atual[0] + delta[0], pos_atual[1] + delta[1])
                percep_pos = {tuple(p.get("pos")): p.get("tipo") for p in percepcoes}
                tipo_na_frente = percep_pos.get(tentativa)
                if tipo_na_frente not in ("recurso", "farol", "ninho"):
                    self.obstaculos_conhecidos.add(tentativa)

    def age(self) -> Accao:
        if self.encontrou_objetivo:
            return Accao("ficar")

        obs = self.ultima_obs
        if obs is None:
            return Accao("ficar")

        pos = tuple(self.pos)
        percepcoes = getattr(obs, "percepcoes", []) or []

        for p in percepcoes:
            if tuple(p.get("pos")) == pos and p.get("tipo") == "ninho":
                self.ultimo_ninho = pos
                if getattr(obs, "carga", 0) > 0:
                    return Accao("depositar")
            elif tuple(p.get("pos")) == pos:
                tipo = p.get("tipo")
                if tipo in ("recurso", "farol") and getattr(obs, "carga", 0) == 0:
                    self.ultimo_recurso = pos
                    quantidade = p.get("quantidade", p.get("valor", 1))
                    if quantidade and quantidade > 0:
                        self.recursos_conhecidos[pos] = quantidade
                    else:
                        self.recursos_conhecidos.pop(pos, None)
                    return Accao("recolher")

        p_tipo = self.politica.get("tipo")
        if p_tipo == "random":
            return politica_aleatoria()
        if p_tipo == "greedy":
            if getattr(obs, "carga", 0) == 0 and self.ultimo_recurso and self.ultimo_recurso in self.recursos_conhecidos:
                return politica_greedy(self, obs, alvo_forcado=self.ultimo_recurso)
            if getattr(obs, "carga", 0) > 0 and self.ultimo_ninho:
                return politica_greedy(self, obs, alvo_forcado=self.ultimo_ninho)
            return politica_greedy(self, obs)

        return Accao("ficar")

    def comunica(self, mensagem: str, de_agente: "Agente"):
        print(f"[{self.nome}] recebeu mensagem de {de_agente.nome}: {mensagem}")
