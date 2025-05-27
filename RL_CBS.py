import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import heapq
from collections import defaultdict, deque
from copy import deepcopy
from typing import List, Tuple, Dict, Set, Optional

# Import RL components
import glob, re
import time
import torch
from RL_train import GridWorld, GridEnv, EnhancedDuelingDQN, DIRS

# --- Types ---
Position   = Tuple[int, int]
Path       = List[Position]
Constraint = Tuple[int, Position]  # (time, cell)

class Agent:
    def __init__(self, start: Position, goal: Position, name: str = ""):
        self.start = start
        self.goal  = goal
        self.name  = name or f"Agent({start}->{goal})"

# Utility to pick the best RL checkpoint (best_*.pth)
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


def plan_path(
    grid: GridWorld,
    start: Position,
    goal: Position,
    constraints: Set[Constraint]
) -> Path:
    """
    RL rollout that *respects* (time, pos) constraints:
    at each step t, picks highest-Q action that DOES NOT land
    in a forbidden (t+1, pos).
    """
    env = GridEnv(grid, start, goal, constraints)
    state = env.reset()
    path = [start]
    done = False
    t = 0

    while not done:
        st_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = model(st_t).cpu().numpy().flatten()

        # sort actions by descending Q
        actions_by_q = sorted(range(len(q_vals)), key=lambda a: -q_vals[a])

        # try each action until one is legal
        for a in actions_by_q:
            # peek next position
            next_pos = (env.pos[0] + DIRS[a][0], env.pos[1] + DIRS[a][1])
            if (t+1, next_pos) in constraints:
                continue
            # if legal, take it
            state, _, done = env.step(a)
            path.append(env.pos)
            break
        else:
            # no legal move -> stuck
            return []

        t += 1

    # pad with goal repeats exactly like A*
    if path[-1] == goal:
        path.extend([goal] * 5)
    return path


def detect_conflict(p1: Path, p2: Path) -> Optional[Tuple[int, Position]]:
    max_len = max(len(p1), len(p2))
    for t in range(max_len):
        a1 = p1[min(t, len(p1)-1)]
        a2 = p2[min(t, len(p2)-1)]
        if a1 == a2:
            return (t, a1)
        if t > 0:
            a1_prev = p1[min(t-1, len(p1)-1)]
            a2_prev = p2[min(t-1, len(p2)-1)]
            if a1 == a2_prev and a2 == a1_prev:
                return (t, a1)
    return None


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


def visualize_paths(grid: GridWorld, paths: Dict[int, Path], save: bool = False, filename: str = "cbs_rl.gif"):
    agent_colors = ['r','g','b','c','m','y','k']
    max_time = max(len(p) for p in paths.values())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-0.5, grid.width-0.5)
    ax.set_ylim(-0.5, grid.height-0.5)
    ax.set_xticks(range(grid.width))
    ax.set_yticks(range(grid.height))
    ax.grid(True)
    for (x,y) in grid.obstacles:
        ax.add_patch(plt.Rectangle((y-0.5, x-0.5),1,1,color='gray'))
    circles = []
    for i in range(len(paths)):
        c = plt.Circle((0,0),0.3,color=agent_colors[i%len(agent_colors)],label=f"A{i}")
        ax.add_patch(c)
        circles.append(c)
    def update(frame):
        for i,path in paths.items():
            x,y = path[min(frame,len(path)-1)]
            circles[i].center = (y, x)
        ax.set_title(f"Step {frame}")
        return circles
    ani = animation.FuncAnimation(fig, update, frames=max_time, interval=500, blit=True)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower center', ncol=len(paths))
    if save:
        ani.save(filename, writer='pillow')
        print(f"Saved animation to {filename}")
    else:
        plt.show()
def cbs_solve(
    grid: GridWorld,
    agents: List[Agent],
    max_expansions: int = 200
) -> Dict[int, Path]:
    class CBSNode:
        def __init__(self, constraints, paths, node_id):
            self.constraints = constraints  # agent_id -> Set[(t,cell)]
            self.paths       = paths        # agent_id -> Path
            self.node_id     = node_id

    def node_cost(n: CBSNode) -> int:
        return sum(len(p) for p in n.paths.values())

    # cache for (start,goal,constraints) -> path
    replan_cache = {}
    def cached_plan(s, g, cons):
        key = (s, g, frozenset(cons))
        if key in replan_cache:
            return replan_cache[key]
        t0 = time.time()
        p = plan_path(grid, s, g, cons)
        print(f"  - Replan {s}->{g} w/ {len(cons)} constraints took {time.time()-t0:.3f}s")
        replan_cache[key] = p
        return p

    root_constraints = defaultdict(set)
    root_paths = {
        i: cached_plan(ag.start, ag.goal, root_constraints[i])
        for i, ag in enumerate(agents)
    }
    root = CBSNode(root_constraints, root_paths, 0)

    open_list = [(node_cost(root), 0, root)]
    heapq.heapify(open_list)

    expansions = 0
    next_id    = 1

    while open_list:
        if expansions >= max_expansions:
            print(f"[!] Reached max expansions ({max_expansions}), aborting.")
            return {}
        expansions += 1

        _, _, node = heapq.heappop(open_list)

        # 1) detect conflict
        conflict = None
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                c = detect_conflict(node.paths[i], node.paths[j])
                if c:
                    conflict = (i, j, c)
                    break
            if conflict:
                break

        # 2) no conflict → done
        if not conflict:
            print(f"[+] Found conflict-free solution after {expansions} expansions.")
            return node.paths

        ai, aj, (t, pos) = conflict
        print(f"- Conflict at t={t}, pos={pos} between A{ai} and A{aj}")

        # 3) branch on each involved agent
        for agent_id in (ai, aj):
            new_cons = deepcopy(node.constraints)
            new_cons[agent_id].add((t, pos))

            new_paths = dict(node.paths)
            p = cached_plan(
                agents[agent_id].start,
                agents[agent_id].goal,
                new_cons[agent_id]
            )
            if not p:
                continue  # infeasible

            new_paths[agent_id] = p
            child = CBSNode(new_cons, new_paths, next_id)
            next_id += 1
            heapq.heappush(open_list, (node_cost(child), child.node_id, child))

    return {}

if __name__ == "__main__":
    import sys
    # Load RL model
    print("[1/5] Finding best RL checkpoint…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_best_checkpoint()
    if ckpt is None:
        raise FileNotFoundError("No RL checkpoint best_*.pth found")
    print(f"[2/5] Loading RL model from {ckpt}")
    model = EnhancedDuelingDQN(8, 8, len(DIRS)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Define grid and agents
    print("[3/5] Setting up grid and agents…")
    grid = GridWorld(8, 8, obstacles={(3,3),(4,4)})
    #agents = [Agent((0,0),(7,7)), Agent((7,0),(0,7)), Agent((0,7),(7,0))]
    agents = [Agent((0,0),(7,7)), Agent((7,0),(0,7)), Agent((0,7),(7,0))]

    # Solve with CBS+RL
    print("[4/5] Running CBS+RL solver… this may take a moment")
    paths = cbs_solve(grid, agents)
    if not paths:
        print("[!] No solution found")
        sys.exit(1)
    print(f"[5/5] Solution found! Max time steps = {max(len(p) for p in paths.values())}")

    # Visualize and save
    print("[*] Rendering and saving animation to cbs_rl.gif…")
    visualize_paths(grid, paths, save=True)
    print("[✓] Done.")

