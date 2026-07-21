import numpy as np
import torch
import torch.nn as nn


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNet(nn.Module):
    def __init__(self, in_ch, h, w, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap, obs_shape, seed=0):
        self.cap = cap
        self.obs = np.zeros((cap, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((cap, *obs_shape), dtype=np.uint8)
        self.action = np.zeros(cap, dtype=np.int64)
        self.reward = np.zeros(cap, dtype=np.float32)
        self.done = np.zeros(cap, dtype=np.float32)
        self.idx = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.cap if self.full else self.idx

    def push(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.idx = (i + 1) % self.cap
        self.full = self.full or self.idx == 0

    def sample(self, n, dev):
        idx = self.rng.integers(0, len(self), n)
        to = lambda a, dt: torch.as_tensor(a, dtype=dt, device=dev)
        return (to(self.obs[idx], torch.float32),
                to(self.action[idx], torch.int64),
                to(self.reward[idx], torch.float32),
                to(self.next_obs[idx], torch.float32),
                to(self.done[idx], torch.float32))


def td_loss(online, target, batch, gamma):
    obs, action, reward, next_obs, done = batch
    q = online(obs).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = target(next_obs).max(1).values
        y = reward + gamma * (1.0 - done) * next_q
    return nn.functional.smooth_l1_loss(q, y)
