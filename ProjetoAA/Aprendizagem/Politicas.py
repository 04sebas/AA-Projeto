import math
import heapq
import random
from Objetos.Accao import Accao

DIRECOES = {
    "cima": (0, -1),
    "baixo": (0, 1),
    "esquerda": (-1, 0),
    "direita": (1, 0),
    "ficar": (0, 0)
}

def soma_pos(p, d):
    return p[0] + d[0], p[1] + d[1]

def distancia(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

def obter_obstaculos(obs):
    return {tuple(p["pos"]) for p in (obs.percepcoes or []) if p.get("tipo") == "obstaculo"}

def obter_ninhos(obs):
    return [tuple(p["pos"]) for p in (obs.percepcoes or []) if p.get("tipo") == "ninho"]

def politica_aleatoria():
    return Accao(random.choice(["cima", "baixo", "esquerda", "direita"]))

def primeira_direcao_livre(pos, obsts, largura=None, altura=None, prefer=None):
    orden = [prefer] if prefer else []
    for d in ["cima", "baixo", "esquerda", "direita"]:
        if d not in orden:
            orden.append(d)
    for nome in orden:
        if not nome:
            continue
        dx, dy = DIRECOES[nome]
        novo = (pos[0] + dx, pos[1] + dy)
        if largura is not None and altura is not None:
            if not (0 <= novo[0] < largura and 0 <= novo[1] < altura):
                continue
        if novo in obsts:
            continue
        return nome
    return "ficar"

def direcao_para_alvo(pos, alvo, obsts, largura=None, altura=None, ultima_direcao=None):
    if pos == alvo:
        return "ficar"
    if largura is None:
        largura = 100
    if altura is None:
        altura = 100

    def heur(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (heur(pos, alvo), 0, pos))
    came_from = {}
    gscore = {pos: 0}
    closed = set()

    while open_heap:
        _, gcur, current = heapq.heappop(open_heap)
        if current == alvo:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            if len(path) >= 2:
                proximo = path[1]
                dx = proximo[0] - pos[0]
                dy = proximo[1] - pos[1]
                for nome, delta in DIRECOES.items():
                    if delta == (dx, dy):
                        return nome
            return "ficar"
        if current in closed:
            continue
        closed.add(current)
        for nome, delta in DIRECOES.items():
            if nome == "ficar":
                continue
            nx = current[0] + delta[0]
            ny = current[1] + delta[1]
            if nx < 0 or ny < 0 or nx >= largura or ny >= altura:
                continue
            neighbor = (nx, ny)
            if neighbor in obsts:
                continue
            tentative_g = gscore[current] + 1
            if tentative_g < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                f = tentative_g + heur(neighbor, alvo)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))
    if ultima_direcao:
        dx, dy = DIRECOES.get(ultima_direcao, (0, 0))
        cand = (pos[0] + dx, pos[1] + dy)
        if 0 <= cand[0] < largura and 0 <= cand[1] < altura and cand not in obsts:
            return ultima_direcao
    return "ficar"

def politica_greedy(agente, obs, alvo_forcado=None):
    pos = tuple(agente.pos)
    obsts = set(obter_obstaculos(obs))
    if hasattr(agente, "obstaculos_conhecidos"):
        obsts |= set(agente.obstaculos_conhecidos)
    largura = getattr(obs, "largura", None)
    altura = getattr(obs, "altura", None)
    carga = getattr(obs, "carga", 0)
    ninhos = [tuple(p["pos"]) for p in (obs.percepcoes or []) if p.get("tipo") == "ninho"]
    recursos_visiveis = [p for p in (obs.percepcoes or []) if p.get("tipo") in ("recurso", "farol")]

    if carga > 0:
        alvo = alvo_forcado or (min(ninhos, key=lambda a: distancia(pos, a)) if ninhos else None)
        if alvo:
            dir_nome = direcao_para_alvo(pos, alvo, obsts, largura, altura, agente.ultima_direcao)
            agente.ultima_direcao = dir_nome
            return Accao(dir_nome)
        if hasattr(obs, "direcao"):
            dx, dy = obs.direcao
            prefer = "direita" if abs(dx) > abs(dy) and dx > 0 else (
                     "esquerda" if abs(dx) > abs(dy) and dx < 0 else (
                     "baixo" if dy > 0 else "cima"))
            dir_nome = prefer if prefer else agente.ultima_direcao
            if dir_nome:
                dx, dy = DIRECOES[dir_nome]
                cand = (pos[0] + dx, pos[1] + dy)
                if 0 <= cand[0] < (largura or 1000) and 0 <= cand[1] < (altura or 1000) and cand not in obsts:
                    agente.ultima_direcao = dir_nome
                    return Accao(dir_nome)
        prefer = None
        if getattr(agente, "ultimo_recurso", None):
            prefer = None
            melhor = -1
            for nome, delta in DIRECOES.items():
                if nome == "ficar":
                    continue
                novo = (pos[0] + delta[0], pos[1] + delta[1])
                if largura is not None and altura is not None:
                    if not (0 <= novo[0] < largura and 0 <= novo[1] < altura):
                        continue
                if novo in obsts:
                    continue
                d = distancia(novo, agente.ultimo_recurso)
                if d > melhor:
                    melhor = d
                    prefer = nome
        dir_nome = primeira_direcao_livre(pos, obsts, largura, altura, prefer)
        agente.ultima_direcao = dir_nome
        return Accao(dir_nome)

    if recursos_visiveis:
        melhor = None
        best_score = -float("inf")
        for r in recursos_visiveis:
            rp = tuple(r["pos"])
            valor = r.get("valor", 1)
            sc = valor / (1 + distancia(pos, rp))
            if sc > best_score:
                best_score = sc
                melhor = rp
        if melhor:
            agente.ultimo_recurso = melhor
            dir_nome = direcao_para_alvo(pos, melhor, obsts, largura, altura, agente.ultima_direcao)
            agente.ultima_direcao = dir_nome
            return Accao(dir_nome)

    if not recursos_visiveis and getattr(agente, "recursos_conhecidos", None):
        melhor = None
        best_score = -float("inf")
        for rp, q in agente.recursos_conhecidos.items():
            sc = q / (1 + distancia(pos, rp))
            if sc > best_score:
                best_score = sc
                melhor = rp
        if melhor:
            agente.ultimo_recurso = melhor
            dir_nome = direcao_para_alvo(pos, melhor, obsts, largura, altura, agente.ultima_direcao)
            agente.ultima_direcao = dir_nome
            return Accao(dir_nome)

    if hasattr(obs, "direcao"):
        dx, dy = obs.direcao
        dir_nome = "direita" if abs(dx) > abs(dy) and dx > 0 else (
                   "esquerda" if abs(dx) > abs(dy) and dx < 0 else (
                   "baixo" if dy > 0 else "cima"))
        agente.ultima_direcao = dir_nome
        return Accao(dir_nome)

    if getattr(agente, "passos_sem_mover", 0) >= getattr(agente, "stuck_threshold", 2):
        dir_nome = primeira_direcao_livre(pos, obsts, largura, altura, agente.ultima_direcao)
        agente.ultima_direcao = dir_nome
        return Accao(dir_nome)

    if agente.ultima_direcao and random.random() < 0.7:
        dx, dy = DIRECOES[agente.ultima_direcao]
        cand = (pos[0] + dx, pos[1] + dy)
        if 0 <= cand[0] < (largura or 1000) and 0 <= cand[1] < (altura or 1000) and cand not in obsts:
            return Accao(agente.ultima_direcao)

    nome = politica_aleatoria()
    agente.ultima_direcao = nome.nome
    return nome


