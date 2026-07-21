import numpy as np

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class Snake:
    n_actions = 3
    n_states = 4 * 8 * 9
    size = 12
    stall_limit = 150

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        c = self.size // 2
        self.body = [(c, c), (c, c - 1), (c, c - 2)]
        self.dir = 1
        self.score = 0
        self.steps_since_food = 0
        self.place_food()
        return self.obs()

    def place_food(self):
        taken = set(self.body)
        free = [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in taken]
        self.food = free[int(self.rng.integers(len(free)))]

    def blocked(self, d):
        dr, dc = DIRS[d]
        r, c = self.body[0]
        r, c = r + dr, c + dc
        if not (0 <= r < self.size and 0 <= c < self.size):
            return True
        return (r, c) in set(self.body[:-1])

    def obs(self):
        d = self.dir
        danger = (int(self.blocked(d)) << 2) \
            | (int(self.blocked((d - 1) % 4)) << 1) \
            | int(self.blocked((d + 1) % 4))
        fr = int(np.sign(self.food[0] - self.body[0][0])) + 1
        fc = int(np.sign(self.food[1] - self.body[0][1])) + 1
        return d * 72 + danger * 9 + fr * 3 + fc

    def step(self, action):
        self.dir = (self.dir + action - 1) % 4
        dr, dc = DIRS[self.dir]
        head = (self.body[0][0] + dr, self.body[0][1] + dc)
        self.steps_since_food += 1
        out = not (0 <= head[0] < self.size and 0 <= head[1] < self.size)
        if out or head in set(self.body[:-1]):
            return self.obs(), -1.0, True
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self.steps_since_food = 0
            if len(self.body) == self.size * self.size:
                return self.obs(), 1.0, True
            self.place_food()
            return self.obs(), 1.0, False
        self.body.pop()
        if self.steps_since_food >= self.stall_limit:
            return self.obs(), -1.0, True
        return self.obs(), -0.01, False

    def frame(self):
        return {"body": [list(seg) for seg in self.body], "food": list(self.food)}


class Pong:
    n_actions = 3
    W = 12
    H = 12
    paddle_w = 3
    max_hits = 30
    n_states = 12 * 12 * 2 * 2 * 10

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.bx = int(self.rng.integers(self.W))
        self.by = 1
        self.vx = 1 if self.rng.random() < 0.5 else -1
        self.vy = 1
        self.px = (self.W - self.paddle_w) // 2
        self.score = 0
        return self.obs()

    def obs(self):
        vx = (self.vx + 1) // 2
        vy = (self.vy + 1) // 2
        return ((self.bx * self.H + self.by) * 4 + vx * 2 + vy) \
            * (self.W - self.paddle_w + 1) + self.px

    def step(self, action):
        self.px = min(max(self.px + action - 1, 0), self.W - self.paddle_w)
        nbx = self.bx + self.vx
        if nbx < 0 or nbx >= self.W:
            self.vx = -self.vx
            nbx = self.bx + self.vx
        nby = self.by + self.vy
        if nby < 0:
            self.vy = -self.vy
            nby = self.by + self.vy
        if nby >= self.H - 1:
            self.bx = nbx
            if self.px <= nbx < self.px + self.paddle_w:
                self.vy = -1
                self.by = self.H - 2
                self.score += 1
                return self.obs(), 1.0, self.score >= self.max_hits
            self.by = self.H - 1
            return self.obs(), -1.0, True
        self.bx, self.by = nbx, nby
        return self.obs(), 0.0, False

    def frame(self):
        return {"ball": [self.bx, self.by], "paddle": self.px}


class Flappy:
    n_actions = 2
    H = 14
    spacing = 10
    gap_half = 2
    max_pipes = 50
    n_states = 10 * 15 * 6

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def rand_gap(self):
        return int(self.rng.integers(3, self.H - 3))

    def reset(self):
        self.y = self.H // 2
        self.v = 0
        self.dist = self.spacing
        self.gap = self.rand_gap()
        self.next_gap = self.rand_gap()
        self.score = 0
        return self.obs()

    def obs(self):
        dy = int(np.clip(self.y - self.gap, -7, 7)) + 7
        v = int(np.clip(self.v, -2, 3)) + 2
        return (self.dist - 1) * 90 + dy * 6 + v

    def step(self, action):
        self.v = -2 if action == 1 else min(self.v + 1, 3)
        self.y += self.v
        self.dist -= 1
        if self.y < 0 or self.y >= self.H:
            self.dist = max(self.dist, 1)
            return self.obs(), -1.0, True
        if self.dist == 0:
            if abs(self.y - self.gap) > self.gap_half:
                self.dist = 1
                return self.obs(), -1.0, True
            self.score += 1
            self.dist = self.spacing
            self.gap = self.next_gap
            self.next_gap = self.rand_gap()
            return self.obs(), 1.0, self.score >= self.max_pipes
        return self.obs(), 0.0, False

    def frame(self):
        return {"y": self.y, "dist": self.dist, "gap": self.gap,
                "next_gap": self.next_gap}
