import math
import heapq
import random

from ProjetoAA.Objects.Action import Action

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "stay": (0, 0)
}

def sum_pos(p, d):
    return p[0] + d[0], p[1] + d[1]

def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

def get_obstacles(obs):
    return {tuple(p["pos"]) for p in (obs.perceptions or []) if p.get("type") == "obstacle"}

def get_nests(obs):
    return [tuple(p["pos"]) for p in (obs.perceptions or []) if p.get("type") == "nest"]

def random_policy():
    return Action(random.choice(["up", "down", "left", "right"]))

def first_free_direction(pos, obstacles, width=None, height=None, prefer=None):
    order = [prefer] if prefer else []
    for d in ["up", "down", "left", "right"]:
        if d not in order:
            order.append(d)
    for name in order:
        if not name:
            continue
        dx, dy = DIRECTIONS[name]
        new_pos = (pos[0] + dx, pos[1] + dy)
        if width is not None and height is not None:
            if not (0 <= new_pos[0] < width and 0 <= new_pos[1] < height):
                continue
        if new_pos in obstacles:
            continue
        return name
    return "stay"

def direction_to_target(pos, target, obstacles, width=None, height=None, last_direction=None):
    if pos == target:
        return "stay"
    if width is None:
        width = 100
    if height is None:
        height = 100

    def heur(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (heur(pos, target), 0, pos))
    came_from = {}
    gscore = {pos: 0}
    closed = set()

    while open_heap:
        _, gcur, current = heapq.heappop(open_heap)
        if current == target:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            if len(path) >= 2:
                next_pos = path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                for name, delta in DIRECTIONS.items():
                    if delta == (dx, dy):
                        return name
            return "stay"
        if current in closed:
            continue
        closed.add(current)
        for name, delta in DIRECTIONS.items():
            if name == "stay":
                continue
            nx = current[0] + delta[0]
            ny = current[1] + delta[1]
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            neighbor = (nx, ny)
            if neighbor in obstacles:
                continue
            tentative_g = gscore[current] + 1
            if tentative_g < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                f = tentative_g + heur(neighbor, target)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))
    if last_direction:
        dx, dy = DIRECTIONS.get(last_direction, (0, 0))
        cand = (pos[0] + dx, pos[1] + dy)
        if 0 <= cand[0] < width and 0 <= cand[1] < height and cand not in obstacles:
            return last_direction
    return "stay"

def greedy_policy(agent, obs, forced_target=None):
    pos = tuple(agent.pos)
    obstacles = set(get_obstacles(obs))
    if hasattr(agent, "known_obstacles"):
        obstacles |= set(agent.known_obstacles)
    width = getattr(obs, "width", None)
    height = getattr(obs, "height", None)
    load = getattr(obs, "load", 0)
    nests = [tuple(p["pos"]) for p in (obs.perceptions or []) if p.get("type") == "nest"]
    visible_resources = [p for p in (obs.perceptions or []) if p.get("type") in ("resource", "beacon")]

    if load > 0:
        target = forced_target or (min(nests, key=lambda a: distance(pos, a)) if nests else None)
        if target:
            dir_name = direction_to_target(pos, target, obstacles, width, height, agent.last_direction)
            agent.last_direction = dir_name
            return Action(dir_name)
        if hasattr(obs, "direction"):
            dx, dy = obs.direction
            prefer = "right" if abs(dx) > abs(dy) and dx > 0 else (
                     "left" if abs(dx) > abs(dy) and dx < 0 else (
                     "down" if dy > 0 else "up"))
            dir_name = prefer if prefer else agent.last_direction
            if dir_name:
                dx, dy = DIRECTIONS[dir_name]
                cand = (pos[0] + dx, pos[1] + dy)
                if 0 <= cand[0] < (width or 100) and 0 <= cand[1] < (height or 100) and cand not in obstacles:
                    agent.last_direction = dir_name
                    return Action(dir_name)
        prefer = None
        if getattr(agent, "last_resource", None):
            prefer = None
            best = -1
            for name, delta in DIRECTIONS.items():
                if name == "stay":
                    continue
                new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                if width is not None and height is not None:
                    if not (0 <= new_pos[0] < width and 0 <= new_pos[1] < height):
                        continue
                if new_pos in obstacles:
                    continue
                d = distance(new_pos, agent.last_resource)
                if d > best:
                    best = d
                    prefer = name
        dir_name = first_free_direction(pos, obstacles, width, height, prefer)
        agent.last_direction = dir_name
        return Action(dir_name)

    if visible_resources:
        best = None
        best_score = -float("inf")
        for r in visible_resources:
            rp = tuple(r["pos"])
            value = r.get("value", 1)
            sc = value / (1 + distance(pos, rp))
            if sc > best_score:
                best_score = sc
                best = rp
        if best:
            agent.last_resource = best
            dir_name = direction_to_target(pos, best, obstacles, width, height, agent.last_direction)
            agent.last_direction = dir_name
            return Action(dir_name)

    if not visible_resources and getattr(agent, "known_resources", None):
        best = None
        best_score = -float("inf")
        for rp, q in agent.known_resources.items():
            sc = q / (1 + distance(pos, rp))
            if sc > best_score:
                best_score = sc
                best = rp
        if best:
            agent.last_resource = best
            dir_name = direction_to_target(pos, best, obstacles, width, height, agent.last_direction)
            agent.last_direction = dir_name
            return Action(dir_name)

    if hasattr(obs, "direction"):
        dx, dy = obs.direction
        dir_name = "right" if abs(dx) > abs(dy) and dx > 0 else (
                   "left" if abs(dx) > abs(dy) and dx < 0 else (
                   "down" if dy > 0 else "up"))
        agent.last_direction = dir_name
        return Action(dir_name)

    if getattr(agent, "steps_without_move", 0) >= getattr(agent, "stuck_threshold", 2):
        dir_name = first_free_direction(pos, obstacles, width, height, agent.last_direction)
        agent.last_direction = dir_name
        return Action(dir_name)

    if agent.last_direction and random.random() < 0.7:
        dx, dy = DIRECTIONS[agent.last_direction]
        cand = (pos[0] + dx, pos[1] + dy)
        if 0 <= cand[0] < (width or 1000) and 0 <= cand[1] < (height or 1000) and cand not in obstacles:
            return Action(agent.last_direction)

    action = random_policy()
    agent.last_direction = action.name
    return action


