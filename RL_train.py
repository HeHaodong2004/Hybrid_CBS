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

Position   = Tuple[int, int]
Constraint = Tuple[int, Position]
DIRS       = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

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

class GridWorld:
    def __init__(self, width: int, height: int, obstacles: Set[Position] = None):
        self.width  = width
        self.height = height
        self.obstacles = obstacles or set()
    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width
    def passable(self, pos: Position) -> bool:
        return pos not in self.obstacles

class GridEnv:
    def __init__(self, grid: GridWorld, start: Position, goal: Position,
                 constraints: Set[Constraint], max_steps: int = 100, gamma: float = 0.99):
        self.grid        = grid
        self.start       = start
        self.goal        = goal
        self.constraints = constraints
        self.max_steps   = max_steps
        self.gamma       = gamma

        self.dist_map, self.max_dist = compute_distance_map(grid, goal)
        self.H, self.W             = grid.height, grid.width
        
        ys, xs = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        self.coord_map = np.stack([ys/self.H, xs/self.W], axis=0).astype(np.float32)
        
        occ = np.ones((self.H, self.W), dtype=np.float32)
        for (r, c) in grid.obstacles:
            occ[r, c] = 0.0
        self.obs_map = occ

    def reset(self):
        self.pos        = self.start
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        ch1 = self.coord_map
        ch2 = self.obs_map[np.newaxis]
        ch3 = (self.dist_map / self.max_dist)[np.newaxis].astype(np.float32)
        ch4 = np.zeros((1, self.H, self.W), dtype=np.float32)
        ch4[0, self.pos[0], self.pos[1]] = 1.0
        ch5 = np.zeros((1, self.H, self.W), dtype=np.float32)
        ch5[0, self.goal[0], self.goal[1]] = 1.0
        state = np.concatenate([ch1, ch2, ch3, ch4, ch5], axis=0)
        return state

    def step(self, action: int):
        old_phi  = -self.dist_map[self.pos[0], self.pos[1]]
        old_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        dr, dc   = DIRS[action]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)
        self.step_count += 1

        reward = -0.3; done = False
        if not self.grid.in_bounds(next_pos) or not self.grid.passable(next_pos) or \
           (self.step_count, next_pos) in self.constraints:
            reward += -1.0; new_phi = old_phi
        else:
            self.pos = next_pos
            new_dist = abs(next_pos[0]-self.goal[0]) + abs(next_pos[1]-self.goal[1])
            reward  += 0.5 * (old_dist - new_dist)
            new_phi  = -self.dist_map[next_pos[0], next_pos[1]]
            if self.pos == self.goal:
                reward += 40.0; done = True
        if self.step_count >= self.max_steps:
            done = True
        shaped = reward + self.gamma * new_phi - old_phi
        return self._get_state(), shaped, done

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return F.relu(self.norm(x + self.net(x)))

class EnhancedDuelingDQN(nn.Module):
    def __init__(self, H, W, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * H * W
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128), nn.LayerNorm(128), nn.ReLU()
        )
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.value_stream = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
        self.adv_stream   = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,output_dim))

    def forward(self, x):
        c = self.conv(x)
        f = self.fc(c)
        f = self.res1(f)
        f = self.res2(f)
        f = self.res3(f)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))

class ReplayBuffer:
    def __init__(self, capacity): self.buffer=deque(maxlen=capacity)
    def push(self, s,a,r,s2,d): self.buffer.append((s,a,r,s2,d))
    def sample(self, batch):
        b = random.sample(self.buffer, batch)
        s,a,r,s2,d = zip(*b)
        return (np.stack(s).astype(np.float32), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2).astype(np.float32), np.array(d, dtype=np.float32))
    def __len__(self): return len(self.buffer)

def train_dqn(width, height, episodes=500, batch_size=64,
              gamma=0.99, lr=1e-3, num_obs=10, tau=0.005, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = EnhancedDuelingDQN(height, width, len(DIRS)).to(device)
    target = EnhancedDuelingDQN(height, width, len(DIRS)).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)
    eps_s, eps_e, eps_d = 1.0, 0.01, episodes/2
    best, recent = -inf, deque(maxlen=50)

    for ep in range(1, episodes+1):
        
        obs_set=set()
        while len(obs_set)<num_obs:
            obs_set.add((random.randrange(height), random.randrange(width)))
        free=[(r,c) for r in range(height) for c in range(width) if (r,c) not in obs_set]
        s, g = random.sample(free, 2)
        env = GridEnv(GridWorld(width, height, obs_set), s, g, set(), 100, gamma)
        state = env.reset()  # state shape: (6, H, W)
        total_r = 0.0
        done    = False

        while not done:
            eps = max(eps_e, eps_s - (eps_s - eps_e) * (ep / eps_d))
            if random.random() < eps:
                action = random.randrange(len(DIRS))
            else:
                with torch.no_grad():
                    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(policy(st).argmax())

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state, total_r = next_state, total_r + reward

            if len(buffer) >= batch_size:
                ss, aa, rr, ss2, dd = buffer.sample(batch_size)
                ss_t  = torch.tensor(ss, dtype=torch.float32, device=device)
                aa_t  = torch.tensor(aa, dtype=torch.long,  device=device).unsqueeze(1)
                rr_t  = torch.tensor(rr, dtype=torch.float32, device=device).unsqueeze(1)
                ss2_t = torch.tensor(ss2, dtype=torch.float32, device=device)
                dd_t  = torch.tensor(dd, dtype=torch.float32, device=device).unsqueeze(1)

                q      = policy(ss_t).gather(1, aa_t)
                next_a = policy(ss2_t).argmax(dim=1, keepdim=True)
                q2     = target(ss2_t).gather(1, next_a)
                target_q = rr_t + gamma * q2 * (1 - dd_t)

                loss = nn.MSELoss()(q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                
                for p, tp in zip(policy.parameters(), target.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * p.data)

        recent.append(total_r)
        if ep % 50 == 0 and len(recent) == 50:
            avg_return = sum(recent) / 50.0
            print(f"Ep {ep}: avg50 = {avg_return:.2f}")
            if avg_return > best:
                best = avg_return
                torch.save(policy.state_dict(), f"best_{avg_return:.2f}.pth")
        if ep % 50 == 0:
            print(f"Ep {ep}/{episodes}, Return = {total_r:.2f}, Eps = {eps:.2f}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width',    type=int, default=8)
    parser.add_argument('--height',   type=int, default=8)
    parser.add_argument('--episodes', type=int, default=15000)
    parser.add_argument('--obstacles',type=int, default=10)
    args = parser.parse_args()
    train_dqn(args.width, args.height, args.episodes, num_obs=args.obstacles)
