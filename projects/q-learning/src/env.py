class CliffWalk:
    rows = 4
    cols = 12
    n_states = 48
    n_actions = 4

    def __init__(self):
        self.start = self.index(3, 0)
        self.goal = self.index(3, 11)
        self.cliff = {self.index(3, c) for c in range(1, 11)}

    @classmethod
    def index(cls, r, c):
        return r * cls.cols + c

    def reset(self):
        return self.start

    def step(self, state, action):
        r, c = divmod(state, self.cols)
        if action == 0:
            r = max(r - 1, 0)
        elif action == 1:
            c = min(c + 1, self.cols - 1)
        elif action == 2:
            r = min(r + 1, self.rows - 1)
        else:
            c = max(c - 1, 0)
        s = self.index(r, c)
        if s in self.cliff:
            return self.start, -100.0, False
        if s == self.goal:
            return s, -1.0, True
        return s, -1.0, False
