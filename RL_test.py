import argparse
import glob
import re
import heapq
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Set, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import RL components
from RL_train import GridWorld, GridEnv, EnhancedDuelingDQN, DIRS

# --- Types ---
Position = Tuple[int, int]
Path = List[Position]
Constraint = Tuple[int, Position]  # (time, cell)

class Agent:
    def __init__(self, start: Position, goal: Position, name: str = ""):
        self.start = start
        self.goal = goal
        self.name = name or f"Agent({start}->{goal})"

# Utility to find best checkpoint

def find_best_checkpoint(pattern="best_*.pth") -> Optional[str]:
    files = glob.glob(pattern)
    best, best_score = None, float('-inf')
    for f in files:
        m = re.search(r"best_([0-9]+(?:\.[0-9]+)?)\.pth$", f)
        if m:
            score = float(m.group(1))
            if score > best_score:
                best_score, best = score, f
    return best

# Replace A* planner with RL planner
def plan_path(grid: GridWorld, start: Position, goal: Position, constraints: Set[Constraint]) -> Path:
    """
    Obtain a path using a trained RL model (Dueling DQN).
    Constraints are not enforced by RL agent currently.
    """
    env = GridEnv(grid, start, goal, constraints)
    state = env.reset()
    path = [start]
    done = False
    while not done:
        st_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = model(st_t)
            action = int(q.argmax(dim=1).item())
        state, _, done = env.step(action)
        path.append(env.pos)
    return path

# Conflict detection

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

# CBS solver

def cbs_solve(grid: GridWorld, agents: List[Agent]) -> Dict[int, Path]:
    class CBSNode:
        def __init__(self, constraints, paths, node_id):
            self.constraints = constraints
            self.paths = paths
            self.node_id = node_id

    def node_cost(n: CBSNode) -> int:
        return sum(len(p) for p in n.paths.values())

    counter = 0
    root_constraints = defaultdict(set)
    root_paths = {}
    for i, ag in enumerate(agents):
        root_paths[i] = plan_path(grid, ag.start, ag.goal, root_constraints[i])
    root = CBSNode(root_constraints, root_paths, counter)

    open_list = [(node_cost(root), root.node_id, root)]
    heapq.heapify(open_list)

    while open_list:
        _, _, node = heapq.heappop(open_list)
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
            return node.paths
        ai, aj, (t, pos) = conflict
        for agent_id in (ai, aj):
            new_cons = deepcopy(node.constraints)
            new_cons[agent_id].add((t, pos))
            new_paths = node.paths.copy()
            p = plan_path(grid, agents[agent_id].start, agents[agent_id].goal, new_cons[agent_id])
            if not p:
                continue
            new_paths[agent_id] = p
            counter += 1
            child = CBSNode(new_cons, new_paths, counter)
            heapq.heappush(open_list, (node_cost(child), child.node_id, child))

    return {}

# Printing and visualization utilities

def print_paths(paths: Dict[int, Path], grid: GridWorld):
    max_time = max(len(p) for p in paths.values())
    for t in range(max_time):
        print(f"Time {t}")
        grid_out = [['.' for _ in range(grid.width)] for _ in range(grid.height)]
        for aid, path in paths.items():
            x, y = path[min(t, len(path)-1)]
            grid_out[x][y] = str(aid)
        for row in grid_out:
            print(' '.join(row))
        print()


def visualize_paths(grid: GridWorld, paths: Dict[int, Path]):
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
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="RL model checkpoint file (default: best_*.pth highest)")
    parser.add_argument("--width", type=int, default=8, help="Grid width")
    parser.add_argument("--height", type=int, default=8, help="Grid height")
    args = parser.parse_args()

    # Determine checkpoint
    ckpt = args.checkpoint or find_best_checkpoint()
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found matching best_*.pth; specify --checkpoint explicitly.")

    # Load RL model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedDuelingDQN(args.height, args.width, len(DIRS)).to(device)
    print(f"Using checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Setup grid and agents
    grid = GridWorld(args.width, args.height, obstacles={(3,3),(4,4)})
    agents = [Agent((0,0),(7,7)), Agent((7,0),(0,7)), Agent((0,7),(7,0))]

    # Run CBS + RL
    paths = cbs_solve(grid, agents)
    if not paths:
        print("No solution found")
    else:
        print_paths(paths, grid)
        visualize_paths(grid, paths)
