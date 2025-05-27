import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import heapq
import glob, re, time, torch
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

class Agent:
    def __init__(
        self,
        start: Position,
        goal: Position,
        planner: str = "astar",
        name: str = ""
    ):
        self.start   = start
        self.goal    = goal
        self.planner = planner  # 'astar' or 'rl'
        self.name    = name or f"Agent({start}->{goal}, {planner})"

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
        _, time_step, current, path = heapq.heappop(open_set)
        if (current, time_step) in visited:
            continue
        visited.add((current, time_step))
        if current == goal:
            return path + [goal] * 5
        for dx, dy in DIRS:
            nbr = (current[0]+dx, current[1]+dy)
            if not (grid.in_bounds(nbr) and grid.passable(nbr)): continue
            if (time_step+1, nbr) in constraints: continue
            new_path = path + [nbr]
            cost = len(new_path) + manhattan(nbr, goal)
            heapq.heappush(open_set, (cost, time_step+1, nbr, new_path))
    return []

# --- RL Planner Wrapper ---
# assumes global `model` and `device`

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
        actions = sorted(range(len(q_vals)), key=lambda a: -q_vals[a])
        for a in actions:
            nx = env.pos[0] + DIRS[a][0]
            ny = env.pos[1] + DIRS[a][1]
            if (t+1, (nx,ny)) in constraints:
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
    else:
        raise ValueError(f"Unknown planner: {method}")

# --- Conflict Detection ---
def detect_conflict(p1: Path, p2: Path) -> Optional[Tuple[int, Position]]:
    max_len = max(len(p1), len(p2))
    for t in range(max_len):
        a1 = p1[min(t, len(p1)-1)]; a2 = p2[min(t, len(p2)-1)]
        if a1 == a2: return (t, a1)
        if t>0 and a1==p2[min(t-1,len(p2)-1)] and a2==p1[min(t-1,len(p1)-1)]:
            return (t, a1)
    return None

# --- CBS Solver ---
def cbs_solve(
    grid: GridWorld,
    agents: List[Agent],
    max_expansions: int = 200
) -> Dict[int, Path]:
    class CBSNode:
        def __init__(self, constraints, paths, node_id):
            self.constraints = constraints
            self.paths       = paths
            self.node_id     = node_id
    def node_cost(n):
        return sum(len(p) for p in n.paths.values())

    # cache replans
    cache = {}
    def cached_plan(ag_id, method, start, goal, cons):
        key = (ag_id, method, start, goal, frozenset(cons))
        if key in cache: return cache[key]
        t0 = time.time()
        p = plan_path(grid, start, goal, cons, method)
        print(f"  - Agent{ag_id} {method} replan w/{len(cons)} cons took {time.time()-t0:.3f}s")
        cache[key] = p
        return p

    # root
    constraints = defaultdict(set)
    paths = {}
    for i, ag in enumerate(agents):
        paths[i] = cached_plan(i, ag.planner, ag.start, ag.goal, constraints[i])
    root = CBSNode(constraints, paths, 0)
    open_list = [(node_cost(root), 0, root)]; heapq.heapify(open_list)

    expansions=0; nid=1
    while open_list:
        if expansions>=max_expansions:
            print("[!] Max expansions reached."); return {}
        expansions+=1
        _,_,node = heapq.heappop(open_list)
        # detect
        conflict=None
        for i in range(len(agents)):
            for j in range(i+1,len(agents)):
                c=detect_conflict(node.paths[i], node.paths[j])
                if c: conflict=(i,j,c); break
            if conflict: break
        if not conflict:
            print(f"[+] Solution after {expansions} expansions."); return node.paths
        ai,aj,(t,pos)=conflict
        print(f"- Conflict t={t},pos={pos} between {ai},{aj}")
        for agent_id in (ai,aj):
            new_cons = deepcopy(node.constraints)
            new_cons[agent_id].add((t,pos))
            new_paths = dict(node.paths)
            ag=agents[agent_id]
            p = cached_plan(agent_id, ag.planner, ag.start, ag.goal, new_cons[agent_id])
            if not p: continue
            new_paths[agent_id]=p
            child=CBSNode(new_cons,new_paths,nid); nid+=1
            heapq.heappush(open_list,(node_cost(child),child.node_id,child))
    return {}

# --- Visualization & Main ---
def visualize_paths(grid, paths, save=False, filename="out.gif"):
    agent_colors=['r','g','b','c','m','y','k']
    max_t=max(len(p) for p in paths.values())
    fig,ax=plt.subplots(figsize=(6,6))
    ax.set_xlim(-.5,grid.width-.5); ax.set_ylim(-.5,grid.height-.5)
    ax.set_xticks(range(grid.width)); ax.set_yticks(range(grid.height))
    ax.grid(True)
    for (x,y) in grid.obstacles: ax.add_patch(plt.Rectangle((y-.5,x-.5),1,1,color='gray'))
    circles=[plt.Circle((0,0),.3,color=agent_colors[i%len(agent_colors)],label=f"A{i}") for i in range(len(paths))]
    for c in circles: ax.add_patch(c)
    def update(f):
        for i,p in paths.items(): circles[i].center=(p[min(f,len(p)-1)][1],p[min(f,len(p)-1)][0])
        ax.set_title(f"Step {f}"); return circles
    ani=animation.FuncAnimation(fig,update,frames=max_t,interval=500,blit=True)
    plt.gca().invert_yaxis(); plt.legend(loc='lower center',ncol=len(paths))
    if save: ani.save(filename,writer='pillow'); print(f"Saved {filename}")
    else: plt.show()

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

if __name__=="__main__":
    # init
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=find_best_checkpoint()
    print(f"Loading RL model {ckpt}")
    model=EnhancedDuelingDQN(8,8,len(DIRS)).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device))
    model.eval()
    # setup
    grid=GridWorld(8,8,obstacles={(3,3),(4,4)})
    # mix planners: agent0 uses astar, agent1 rl, agent2 astar
    agents=[Agent((0,0),(7,7),'astar'),Agent((7,0),(0,7),'rl'),Agent((0,7),(7,0),'astar')]
    # solve
    paths=cbs_solve(grid,agents)
    if not paths: print("No solution"); exit(1)
    print_paths(paths,grid)
    visualize_paths(grid,paths,save=True,filename="mixed_cbs.gif")
