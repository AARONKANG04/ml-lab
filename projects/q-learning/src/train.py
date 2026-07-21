import argparse
import json
from pathlib import Path

import numpy as np

from env import CliffWalk

METHODS = ["q_learning", "sarsa", "expected_sarsa"]


def eps_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[rng.integers(len(best))])


def expected_value(q_row, epsilon):
    best = np.flatnonzero(q_row == q_row.max())
    probs = np.full(len(q_row), epsilon / len(q_row))
    probs[best] += (1.0 - epsilon) / len(best)
    return float(probs @ q_row)


def run(method, env, episodes, epsilon, lr, gamma, seed, max_steps=400):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(episodes)

    for ep in range(episodes):
        s = env.reset()
        a = eps_greedy(q[s], epsilon, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(s, a)
            total += r
            if done:
                target = r
            elif method == "q_learning":
                target = r + gamma * q[s2].max()
            elif method == "sarsa":
                a2 = eps_greedy(q[s2], epsilon, rng)
                target = r + gamma * q[s2, a2]
            else:
                target = r + gamma * expected_value(q[s2], epsilon)
            q[s, a] += lr * (target - q[s, a])
            if done:
                break
            s = s2
            a = a2 if method == "sarsa" else eps_greedy(q[s], epsilon, rng)
        returns[ep] = total
    return q, returns


def greedy_rollout(q, env, max_steps=100):
    s = env.reset()
    path = [s]
    total = 0.0
    for _ in range(max_steps):
        a = int(q[s].argmax())
        s, r, done = env.step(s, a)
        total += r
        path.append(s)
        if done:
            return total, path, True
    return total, path, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--sweep-seeds", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out", default="assets/results.json")
    args = ap.parse_args()

    env = CliffWalk()
    results = {"config": vars(args), "methods": {}}

    for method in METHODS:
        curves, greedy_returns, reached = [], [], 0
        path0 = None
        for seed in range(args.seeds):
            q, returns = run(method, env, args.episodes, args.epsilon,
                             args.lr, args.gamma, seed)
            curves.append(returns)
            g, path, done = greedy_rollout(q, env)
            greedy_returns.append(g)
            reached += int(done)
            if seed == 0:
                path0 = path
        curves = np.stack(curves)
        results["methods"][method] = {
            "online_return": curves.mean(0).tolist(),
            "online_last100": float(curves[:, -100:].mean()),
            "greedy_return_mean": float(np.mean(greedy_returns)),
            "greedy_reached_goal": reached,
            "greedy_path_seed0": [list(divmod(s, env.cols)) for s in path0],
        }
        print(f"{method:14s} online last100 {curves[:, -100:].mean():7.2f}   "
              f"greedy {np.mean(greedy_returns):7.2f}   "
              f"reached goal {reached}/{args.seeds}")

    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    sweep = {m: [] for m in METHODS}
    for eps in epsilons:
        for method in METHODS:
            vals = []
            for seed in range(args.sweep_seeds):
                _, returns = run(method, env, args.episodes, eps,
                                 args.lr, args.gamma, 1000 + seed)
                vals.append(returns[-100:].mean())
            sweep[method].append(float(np.mean(vals)))
        print(f"eps {eps:.2f}  " + "  ".join(
            f"{m} {sweep[m][-1]:7.2f}" for m in METHODS))
    results["sweep"] = {"epsilons": epsilons, **sweep}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
