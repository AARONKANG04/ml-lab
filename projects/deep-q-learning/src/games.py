import numpy as np

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class Snake:
    n_actions = 4
    obs_channels = 3
    H = 12
    W = 12
    stall_limit = 200

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        c = self.W // 2
        self.body = [(c, c), (c, c - 1), (c, c - 2)]
        self.dir = 1
        self.score = 0
        self.steps_since_food = 0
        self.place_food()
        return self.obs()

    def place_food(self):
        taken = set(self.body)
        free = [(r, c) for r in range(self.H) for c in range(self.W)
                if (r, c) not in taken]
        self.food = free[int(self.rng.integers(len(free)))]

    def obs(self):
        o = np.zeros((3, self.H, self.W), dtype=np.uint8)
        o[0][self.body[0]] = 1
        for seg in self.body[1:]:
            o[1][seg] = 1
        o[2][self.food] = 1
        return o

    def step(self, action):
        if (action + 2) % 4 != self.dir:
            self.dir = action
        dr, dc = DIRS[self.dir]
        head = (self.body[0][0] + dr, self.body[0][1] + dc)
        self.steps_since_food += 1
        out = not (0 <= head[0] < self.H and 0 <= head[1] < self.W)
        if out or head in set(self.body[:-1]):
            return self.obs(), -1.0, True
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self.steps_since_food = 0
            if len(self.body) == self.H * self.W:
                return self.obs(), 1.0, True
            self.place_food()
            return self.obs(), 1.0, False
        self.body.pop()
        if self.steps_since_food >= self.stall_limit:
            return self.obs(), -1.0, True
        return self.obs(), 0.0, False

    def frame(self):
        return {"body": [list(seg) for seg in self.body], "food": list(self.food)}


class Breakout:
    n_actions = 3
    obs_channels = 3
    H = 14
    W = 12
    paddle_w = 3
    n_bricks = 48
    max_steps = 3000

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.bricks = np.ones((4, self.W), dtype=np.uint8)
        self.bx = int(self.rng.integers(self.W))
        self.by = 7
        self.vx = 1 if self.rng.random() < 0.5 else -1
        self.vy = 1
        self.px = (self.W - self.paddle_w) // 2
        self.score = 0
        self.t = 0
        return self.obs()

    def obs(self):
        o = np.zeros((3, self.H, self.W), dtype=np.uint8)
        o[0][1:5] = self.bricks
        o[1][self.by, self.bx] = 1
        o[2][self.H - 1, self.px:self.px + self.paddle_w] = 1
        return o

    def step(self, action):
        self.t += 1
        self.px = min(max(self.px + action - 1, 0), self.W - self.paddle_w)
        reward = 0.0

        nbx = self.bx + self.vx
        if nbx < 0 or nbx >= self.W:
            self.vx = -self.vx
            nbx = self.bx + self.vx
        nby = self.by + self.vy
        if nby < 0:
            self.vy = -self.vy
            nby = self.by + self.vy

        if 1 <= nby <= 4 and self.bricks[nby - 1, nbx]:
            self.bricks[nby - 1, nbx] = 0
            self.score += 1
            reward = 1.0
            self.vy = -self.vy
            self.bx, self.by = nbx, nby
            if self.bricks.sum() == 0:
                return self.obs(), reward, True
            return self.obs(), reward, False

        if nby >= self.H - 1:
            self.bx = nbx
            if self.px <= nbx < self.px + self.paddle_w:
                self.vy = -1
                self.by = self.H - 2
                return self.obs(), reward, self.t >= self.max_steps
            self.by = self.H - 1
            return self.obs(), -1.0, True

        self.bx, self.by = nbx, nby
        return self.obs(), reward, self.t >= self.max_steps

    def frame(self):
        return {"ball": [self.bx, self.by], "paddle": self.px,
                "bricks": self.bricks.copy().tolist()}
