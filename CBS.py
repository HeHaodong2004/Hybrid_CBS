import random
import heapq
import glob
import re
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import defaultdict, deque
from copy import deepcopy
from typing import List, Tuple, Dict, Set, Optional, Callable

# Import RL components
from RL_train import GridWorld, GridEnv, EnhancedDuelingDQN, DIRS

# --- Types ---
Position   = Tuple[int, int]
Path       = List[Position]
Constraint = Tuple[int, Position]  # (time, cell)
Planner    = Callable[[Path, Path, Set[Constraint]], Path]

# --- A* Planner Wrapper ---
def astar_plan(
    grid: GridWorld,
    start: Position,
    goal: Position,
    constraints: Set[Constraint]
) -> Path:
    from math import inf
    def manhattan(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    open_set = [(manhattan(start, goal), 0, start, [start])]
    visited  = set()
    while open_set:
        _, t, curr, path = heapq.heappop(open_set)
        if (curr, t) in visited:
            continue
        visited.add((curr, t))
        if curr == goal:
            return path + [goal]*5
        for dx, dy in DIRS:
            nbr = (curr[0]+dx, curr[1]+dy)
            if not (grid.in_bounds(nbr) and grid.passable(nbr)):
                continue
            if (t+1, nbr) in constraints:
                continue
            new_path = path + [nbr]
            cost = len(new_path) + manhattan(nbr, goal)
            heapq.heappush(open_set, (cost, t+1, nbr, new_path))
    return []

# --- RL Planner Wrapper ---
def find_best_checkpoint(pattern: str = "best_*.pth") -> Optional[str]:
    best_ckpt = None
    best_score = float('-inf')
    for f in glob.glob(pattern):
        m = re.search(r"best_([0-9]+(?:\.[0-9]+)?)\.pth$", f)
        if m:
            score = float(m.group(1))
            if score > best_score:
                best_score, best_ckpt = score, f
    return best_ckpt

def rl_plan(
    grid: GridWorld,
    start: Position,
    goal: Position,
    constraints: Set[Constraint]
) -> Path:
    env = GridEnv(grid, start, goal, constraints)
    state = env.reset()
    path = [start]
    done = False
    t = 0
    while not done:
        st_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = model(st_t).cpu().numpy().flatten()
        # try actions in descending Q‐value order
        for a in sorted(range(len(q_vals)), key=lambda i: -q_vals[i]):
            nx = env.pos[0] + DIRS[a][0]
            ny = env.pos[1] + DIRS[a][1]
            if (t+1, (nx, ny)) in constraints:
                continue
            state, _, done = env.step(a)
            path.append(env.pos)
            break
        else:
            return []
        t += 1
    if path[-1] == goal:
        path.extend([goal]*5)
    return path

   
# --- RRT* (no diagonals) + A* fallback ---
def rrt_star_plan(
    grid: GridWorld,
    start: Position,
    goal: Position,
    constraints: Set[Constraint],
    max_iter: int = 200
) -> Path:
    from math import inf

    def neighbors(pos: Position):
        for dx, dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            yield (pos[0]+dx, pos[1]+dy)

    if start == goal:
        return [start] + [goal]*5

    parents = {start: None}
    times   = {start: 0}
    frontier = [start]

    for it in range(max_iter):

        curr = random.choice(frontier)

        nbrs = list(neighbors(curr))
        random.shuffle(nbrs)
        for new in nbrs:
            # only cardinal & stay
            if not grid.in_bounds(new) or not grid.passable(new):
                continue
            t_new = times[curr] + 1
            if (t_new, new) in constraints:
                continue

            if new in parents:
                # already visited
                continue
            parents[new] = curr
            times[new]   = t_new
            frontier.append(new)

            if new == goal:
                # rebuild path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parents[node]
                path.reverse()
                return path + [goal]*5
            break  
    return astar_plan(grid, start, goal, constraints)


# --- Dispatcher ---
def plan_path(
    grid: GridWorld,
    start: Position,
    goal: Position,
    constraints: Set[Constraint],
    method: str
) -> Path:
    if method == 'astar':
        return astar_plan(grid, start, goal, constraints)
    elif method == 'rl':
        return rl_plan(grid, start, goal, constraints)
    elif method == 'rrt':
        return rrt_star_plan(grid, start, goal, constraints)
    else:
        raise ValueError(f"Unknown planner: {method}")

# --- Conflict Detection ---
def detect_conflict(p1: Path, p2: Path) -> Optional[Tuple[int, Position]]:
    max_len = max(len(p1), len(p2))
    for t in range(max_len):
        a1 = p1[min(t, len(p1)-1)]
        a2 = p2[min(t, len(p2)-1)]
        if a1 == a2:
            return (t, a1)
        # edge‐swap conflict
        if t > 0 and a1 == p2[min(t-1, len(p2)-1)] and a2 == p1[min(t-1, len(p1)-1)]:
            return (t, a1)
    return None

# --- CBS Solver ---
def cbs_solve(
    grid: GridWorld,
    agents: List["Agent"],
    max_expansions: int = 200
) -> Dict[int, Path]:
    class CBSNode:
        def __init__(self, constraints, paths, node_id):
            self.constraints = constraints
            self.paths       = paths
            self.node_id     = node_id

    def node_cost(n: CBSNode) -> int:
        return sum(len(p) for p in n.paths.values())

    cache = {}
    def cached_plan(aid, method, start, goal, cons):
        key = (aid, method, start, goal, frozenset(cons))
        if key in cache:
            return cache[key]
        t0 = time.time()
        p = plan_path(grid, start, goal, cons, method)
        print(f"  - Agent{aid} {method} replan w/{len(cons)} cons took {time.time()-t0:.3f}s")
        cache[key] = p
        return p

    # root
    constraints = defaultdict(set)
    paths = {}
    for i, ag in enumerate(agents):
        paths[i] = cached_plan(i, ag.planner, ag.start, ag.goal, constraints[i])
    root = CBSNode(constraints, paths, 0)
    open_list = [(node_cost(root), 0, root)]
    heapq.heapify(open_list)

    expansions, nid = 0, 1
    while open_list:
        if expansions >= max_expansions:
            print("[!] Max expansions reached.")
            return {}
        expansions += 1
        _, _, node = heapq.heappop(open_list)

        # find a conflict
        conflict = None
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                c = detect_conflict(node.paths[i], node.paths[j])
                if c:
                    conflict = (i, j, c)
                    break
            if conflict:
                break

        if not conflict:
            print(f"[+] Solution after {expansions} expansions.")
            return node.paths

        ai, aj, (t_conf, pos_conf) = conflict
        print(f"- Conflict t={t_conf},pos={pos_conf} between {ai},{aj}")
        for agent_id in (ai, aj):
            new_cons = deepcopy(node.constraints)
            new_cons[agent_id].add((t_conf, pos_conf))
            new_paths = dict(node.paths)
            ag = agents[agent_id]
            p = cached_plan(agent_id, ag.planner, ag.start, ag.goal, new_cons[agent_id])
            if not p:
                continue
            new_paths[agent_id] = p
            child = CBSNode(new_cons, new_paths, nid)
            nid += 1
            heapq.heappush(open_list, (node_cost(child), child.node_id, child))

    return {}

# --- Visualization & I/O ---
def visualize_paths(grid, paths, save=False, filename="out.gif"):
    agent_colors = ['r','g','b','c','m','y','k']
    max_t = max(len(p) for p in paths.values())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-.5, grid.width-.5)
    ax.set_ylim(-.5, grid.height-.5)
    ax.set_xticks(range(grid.width))
    ax.set_yticks(range(grid.height))
    ax.grid(True)
    # draw obstacles
    for (x,y) in grid.obstacles:
        ax.add_patch(plt.Rectangle((y-.5, x-.5), 1, 1, color='gray'))
    circles = [
        plt.Circle((0, 0), .3, color=agent_colors[i%len(agent_colors)], label=f"A{i}")
        for i in paths
    ]
    for c in circles:
        ax.add_patch(c)

    def update(frame):
        for i, p in paths.items():
            x, y = p[min(frame, len(p)-1)]
            circles[i].center = (y, x)
        ax.set_title(f"Step {frame}")
        return circles

    ani = animation.FuncAnimation(fig, update, frames=max_t, interval=500, blit=True)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower center', ncol=len(paths))
    if save:
        ani.save(filename, writer='pillow')
        print(f"Saved {filename}")
    else:
        plt.show()

def print_paths(paths: Dict[int, Path], grid: GridWorld):
    max_time = max(len(p) for p in paths.values())
    for t in range(max_time):
        print(f"Time {t}")
        grid_disp = [['.' for _ in range(grid.width)] for _ in range(grid.height)]
        for aid, path in paths.items():
            x, y = path[min(t, len(path)-1)]
            grid_disp[x][y] = str(aid)
        for row in grid_disp:
            print(' '.join(row))
        print()

class Agent:
    def __init__(self, start: Position, goal: Position, planner: str = "astar", name: str = ""):
        self.start   = start
        self.goal    = goal
        self.planner = planner
        self.name    = name or f"Agent({start}->{goal}, {planner})"

if __name__ == "__main__":
    # load RL model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_best_checkpoint()
    print(f"Loading RL model {ckpt}")
    model = EnhancedDuelingDQN(8, 8, len(DIRS)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # define grid and agents
    grid = GridWorld(8, 8, obstacles={(3,3), (4,4)})
    agents = [
        Agent((0,1), (7,7), 'rrt'),
        Agent((5,0), (2,5), 'rrt'),
        Agent((0,5), (7,6), 'rl'),
        Agent((3,5), (2,6), 'astar'),
        Agent((0,6), (2,3), 'rl'),
    ]

    # solve and output
    paths = cbs_solve(grid, agents)
    if not paths:
        print("No solution")
        exit(1)
    print_paths(paths, grid)
    visualize_paths(grid, paths, save=True, filename="mixed_cbs_rrt.gif")
