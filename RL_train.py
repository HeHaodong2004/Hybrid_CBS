import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Set
from math import inf
import torch.nn.functional as F
from collections import deque as _deque

# -------------------------------
# Type and Constant Definitions
# -------------------------------
Position = Tuple[int, int]
Constraint = Tuple[int, Position]
DIRS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # Five actions: stay, up, down, left, right

# -------------------------------
# Shortest Distance Map Computation
# -------------------------------
def compute_distance_map(grid, goal: Position) -> np.ndarray:
    h, w = grid.height, grid.width
    max_dist = h + w
    dist = np.full((h, w), np.inf, dtype=np.float32)
    dq = _deque()
    dist[goal[0], goal[1]] = 0
    dq.append(goal)
    while dq:
        r, c = dq.popleft()
        for dr, dc in DIRS[1:]:
            nr, nc = r + dr, c + dc
            if grid.in_bounds((nr, nc)) and grid.passable((nr, nc)) and dist[nr, nc] == np.inf:
                dist[nr, nc] = dist[r, c] + 1
                dq.append((nr, nc))
    dist[np.isinf(dist)] = max_dist
    return dist, max_dist

# -------------------------------
# GridWorld Environment Base Class
# -------------------------------
class GridWorld:
    def __init__(self, width: int, height: int, obstacles: Set[Position] = None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def passable(self, pos: Position) -> bool:
        return pos not in self.obstacles

# -------------------------------
# RL Environment Wrapper
# -------------------------------
class GridEnv:
    def __init__(self, grid: GridWorld, start: Position, goal: Position,
                 constraints: Set[Constraint], max_steps: int = 100, gamma: float = 0.99):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.constraints = constraints
        self.max_steps = max_steps
        self.gamma = gamma

        # Compute the shortest distance from each cell to the goal, used as part of the shaping reward
        self.dist_map, self.max_dist = compute_distance_map(grid, goal)
        self.H, self.W = grid.height, grid.width

        # Coordinate normalization channels (2, H, W): first channel is row/H, second channel is col/W
        ys, xs = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        self.coord_map = np.stack([ys / self.H, xs / self.W], axis=0).astype(np.float32)

        # Obstacle binary channel (1, H, W)
        occ = np.ones((self.H, self.W), dtype=np.float32)
        for (r, c) in grid.obstacles:
            occ[r, c] = 0.0
        self.obs_map = occ

    def reset(self):
        self.pos = self.start
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        # state shape: (6, H, W)
        # ch1: coordinate normalization (2, H, W)
        ch1 = self.coord_map
        # ch2: obstacles (1, H, W)
        ch2 = self.obs_map[np.newaxis]
        # ch3: shortest distance normalization (1, H, W)
        ch3 = (self.dist_map / self.max_dist)[np.newaxis].astype(np.float32)
        # ch4: current position one-hot (1, H, W)
        ch4 = np.zeros((1, self.H, self.W), dtype=np.float32)
        ch4[0, self.pos[0], self.pos[1]] = 1.0
        # ch5: goal position one-hot (1, H, W)
        ch5 = np.zeros((1, self.H, self.W), dtype=np.float32)
        ch5[0, self.goal[0], self.goal[1]] = 1.0

        state = np.concatenate([ch1, ch2, ch3, ch4, ch5], axis=0)
        return state

    def step(self, action: int):
        old_phi = -self.dist_map[self.pos[0], self.pos[1]]
        old_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        dr, dc = DIRS[action]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)
        self.step_count += 1

        reward = -0.3
        done = False
        if not self.grid.in_bounds(next_pos) or not self.grid.passable(next_pos) or \
           (self.step_count, next_pos) in self.constraints:
            # Hitting wall, out of bounds or colliding with a constraint, stay in place
            reward += -1.0
            new_phi = old_phi
        else:
            self.pos = next_pos
            new_dist = abs(next_pos[0] - self.goal[0]) + abs(next_pos[1] - self.goal[1])
            reward += 0.5 * (old_dist - new_dist)
            new_phi = -self.dist_map[next_pos[0], next_pos[1]]
            if self.pos == self.goal:
                reward += 40.0
                done = True

        if self.step_count >= self.max_steps:
            done = True

        # Shaping reward = reward + γ * new_phi - old_phi
        shaped = reward + self.gamma * new_phi - old_phi
        return self._get_state(), shaped, done

# -------------------------------
# Dueling DQN + Residual Block Network Definition
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return F.relu(self.norm(x + self.net(x)))

class EnhancedDuelingDQN(nn.Module):
    def __init__(self, H, W, output_dim):
        super().__init__()
        # Input channels = 6 (coordinates 2 + obstacle 1 + distance 1 + current pos 1 + goal pos 1)
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * H * W

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Three residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)

        # value-stream and advantage-stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, 6, H, W)
        c = self.conv(x)          # (batch_size, 64 * H * W)
        f = self.fc(c)            # (batch_size, 128)
        f = self.res1(f)          # (batch_size, 128)
        f = self.res2(f)          # (batch_size, 128)
        f = self.res3(f)          # (batch_size, 128)
        v = self.value_stream(f)  # (batch_size, 1)
        a = self.adv_stream(f)    # (batch_size, output_dim)
        return v + (a - a.mean(dim=1, keepdim=True))

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        # s, s2: np.ndarray((6, H, W))
        # a: int, r: float, d: bool
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s).astype(np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.stack(s2).astype(np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# -------------------------------
# Parallel DQN Training Function (with checkpoint saving)
# -------------------------------
def train_dqn_parallel(width: int,
                       height: int,
                       episodes: int = 500,
                       batch_size: int = 64,
                       gamma: float = 0.99,
                       lr: float = 1e-3,
                       num_obs: int = 10,
                       tau: float = 0.005,
                       K: int = 16,
                       seed: int = 42):
    """
    Run K environments in parallel, each step generates K experiences, and updates the network together.
    Checkpoint saving condition: the average reward of the latest 100 episodes reaches a historical high.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Initialize policy_net and target_net
    policy_net = EnhancedDuelingDQN(height, width, len(DIRS)).to(device)
    target_net = EnhancedDuelingDQN(height, width, len(DIRS)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # 2) Initialize ReplayBuffer
    buffer = ReplayBuffer(capacity=100000)

    # ε decay strategy parameters
    eps_start, eps_end = 1.0, 0.01
    # Roughly estimate the global step limit when parallel (only for eps decay)
    eps_decay_steps = episodes * (width * height) // K
    global_step = 0

    # 3) Construct K parallel environment instances, and randomly generate obstacles, start, and goal
    envs = []
    for _ in range(K):
        obs_set = set()
        while len(obs_set) < num_obs:
            obs_set.add((random.randrange(height), random.randrange(width)))
        free_cells = [(r, c) for r in range(height) for c in range(width) if (r, c) not in obs_set]
        s, g = random.sample(free_cells, 2)
        env = GridEnv(GridWorld(width, height, obs_set), s, g, set(), max_steps=100, gamma=gamma)
        envs.append(env)

    # 4) Parallel reset, get states: list[np.ndarray((6, H, W))] length = K
    states = [env.reset() for env in envs]
    states_batch = np.stack(states, axis=0)  # (K, 6, H, W)
    states_t = torch.tensor(states_batch, device=device, dtype=torch.float32)

    # Count the number of completed full episodes
    finished_episodes = 0

    # Maintain the current accumulated return for each environment
    episode_returns = [0.0 for _ in range(K)]
    # Recent 100 episode returns for averaging
    recent_returns = deque(maxlen=100)
    best_avg = -inf  # Best average over last 100 episodes

    # 5) Main loop: keep sampling in parallel until enough episodes are completed
    while finished_episodes < episodes:
        global_step += 1

        # 5.1) Calculate current ε
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (global_step / eps_decay_steps))

        # 5.2) Use policy_net to infer K states at once → get (K, |DIRS|) Q-values
        with torch.no_grad():
            q_values = policy_net(states_t)               # (K, |DIRS|)
            greedy_actions = q_values.argmax(dim=1)       # (K,)

        # 5.3) ε-greedy generate K actions
        actions = np.zeros((K,), dtype=np.int64)
        for k in range(K):
            if random.random() < eps:
                actions[k] = random.randrange(len(DIRS))
            else:
                actions[k] = int(greedy_actions[k].item())

        # 5.4) Step K environments in parallel
        next_states, rewards, dones = [], [], []
        for k in range(K):
            ns, r, d = envs[k].step(int(actions[k]))
            # Accumulate the return for this environment
            episode_returns[k] += r
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)

            if d:
                # If k-th environment is done, completed a full episode
                finished_episodes += 1
                ret = episode_returns[k]
                recent_returns.append(ret)
                episode_returns[k] = 0.0  # Reset the accumulated return for this env

                # Calculate the average of the latest 100 returns
                if len(recent_returns) == 100:
                    avg_recent = sum(recent_returns) / 100.0
                    if avg_recent > best_avg:
                        best_avg = avg_recent
                        ckpt_name = f"best_{best_avg:.2f}.pth"
                        torch.save(policy_net.state_dict(), ckpt_name)
                        print(f"New best average over 100 episodes: {best_avg:.2f}, saved checkpoint {ckpt_name}")

                # Reset the environment, start a new episode
                ns = envs[k].reset()
                next_states[-1] = ns

        # 5.5) Push K (s, a, r, s2, done) transitions into the same Buffer
        for k in range(K):
            buffer.push(states[k], int(actions[k]), float(rewards[k]), next_states[k], bool(dones[k]))

        # 5.6) If Buffer is big enough, sample a batch of experiences to update the network
        if len(buffer) >= batch_size:
            ss, aa, rr, ss2, dd = buffer.sample(batch_size)
            ss_t  = torch.tensor(ss,  device=device, dtype=torch.float32)  # (batch_size, 6, H, W)
            aa_t  = torch.tensor(aa,  device=device, dtype=torch.long).unsqueeze(1)    # (batch_size, 1)
            rr_t  = torch.tensor(rr,  device=device, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1)
            ss2_t = torch.tensor(ss2, device=device, dtype=torch.float32)  # (batch_size, 6, H, W)
            dd_t  = torch.tensor(dd,  device=device, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1)

            # Double DQN update
            q_values_sel = policy_net(ss_t).gather(1, aa_t)                    # (batch_size, 1)
            next_actions = policy_net(ss2_t).argmax(dim=1, keepdim=True)       # (batch_size, 1)
            q_next = target_net(ss2_t).gather(1, next_actions)                 # (batch_size, 1)
            target_q = rr_t + gamma * q_next * (1 - dd_t)                      # (batch_size, 1)
            loss = nn.MSELoss()(q_values_sel, target_q.detach())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

            # Soft update target_net
            for p, tp in zip(policy_net.parameters(), target_net.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(tau * p.data)

        # 5.7) Update states ← next_states, for next loop
        states = next_states
        states_batch = np.stack(states, axis=0)  # (K, 6, H, W)
        states_t = torch.tensor(states_batch, device=device, dtype=torch.float32)

        # 5.8) Simple progress print
        if global_step % 1000 == 0 or finished_episodes >= episodes:
            print(f"[GlobalStep {global_step}] Finished episodes: {finished_episodes}/{episodes}, "
                  f"Buffer size: {len(buffer)}, Eps: {eps:.3f}, BestAvg100: {best_avg:.2f}")

    print("Parallel training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel DQN Training on GridEnv")
    parser.add_argument('--width', type=int, default=8, help='Grid width')
    parser.add_argument('--height', type=int, default=8, help='Grid height')
    parser.add_argument('--episodes', type=int, default=15000, help='Total number of episodes to complete')
    parser.add_argument('--obstacles', type=int, default=10, help='Number of obstacles per environment')
    parser.add_argument('--batch_size', type=int, default=64, help='Replay buffer sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor γ')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update factor for target network')
    parser.add_argument('--num_obs', type=int, default=10, help='Number of obstacles (num_obs)')
    parser.add_argument('--parallel_envs', type=int, default=16, help='Number of parallel environments K')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Call the parallel training function
    train_dqn_parallel(width=args.width,
                       height=args.height,
                       episodes=args.episodes,
                       batch_size=args.batch_size,
                       gamma=args.gamma,
                       lr=args.lr,
                       num_obs=args.obstacles,
                       tau=args.tau,
                       K=args.parallel_envs,
                       seed=args.seed)
