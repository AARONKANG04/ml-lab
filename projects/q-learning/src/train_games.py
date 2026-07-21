import argparse
import json
from pathlib import Path

import numpy as np

from games import Flappy, Pong, Snake

CONFIGS = {
    "snake": {"cls": Snake, "episodes": 30000, "gamma": 0.9, "lr": 0.1},
    "pong": {"cls": Pong, "episodes": 5000, "gamma": 0.95, "lr": 0.1},
    "flappy": {"cls": Flappy, "episodes": 20000, "gamma": 0.95, "lr": 0.1},
}


def epsilon_at(ep, episodes, floor=0.02):
    return max(floor, 1.0 - ep / (0.6 * episodes))


def train(cls, episodes, gamma, lr, seed, max_steps=2000):
    env = cls(seed=seed)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    scores = np.zeros(episodes)
    for ep in range(episodes):
        eps = epsilon_at(ep, episodes)
        s = env.reset()
        for _ in range(max_steps):
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = int(q[s].argmax())
            s2, r, done = env.step(a)
            target = r if done else r + gamma * q[s2].max()
            q[s, a] += lr * (target - q[s, a])
            s = s2
            if done:
                break
        scores[ep] = env.score
    return q, scores


def greedy_score(cls, q, seed, n=30, max_steps=2000):
    out = []
    for i in range(n):
        env = cls(seed=10000 + seed * 100 + i)
        s = env.reset()
        for _ in range(max_steps):
            s, _, done = env.step(int(q[s].argmax()))
            if done:
                break
        out.append(env.score)
    return out


def record_replay(cls, q, seed, max_frames=240):
    env = cls(seed=seed)
    s = env.reset()
    frames = [env.frame()]
    for _ in range(max_frames - 1):
        s, _, done = env.step(int(q[s].argmax()))
        frames.append(env.frame())
        if done:
            break
    return {"frames": frames, "score": env.score}


def bucket(mean_curve, points=300):
    k = max(1, len(mean_curve) // points)
    return {
        "every": k,
        "scores": [float(np.mean(mean_curve[i:i + k]))
                   for i in range(0, len(mean_curve), k)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="assets/games.json")
    args = ap.parse_args()

    results = {}
    for name, cfg in CONFIGS.items():
        cls = cfg["cls"]
        all_scores, greedy, best_q, best_mean = [], [], None, -1e9
        for seed in range(args.seeds):
            q, scores = train(cls, cfg["episodes"], cfg["gamma"], cfg["lr"], seed)
            g = greedy_score(cls, q, seed)
            all_scores.append(scores)
            greedy.append(g)
            if np.mean(g) > best_mean:
                best_mean, best_q, best_seed = np.mean(g), q, seed
            print(f"{name} seed {seed}: train last500 "
                  f"{scores[-500:].mean():6.2f}  greedy {np.mean(g):6.2f} "
                  f"(max {max(g)})")
        mean_curve = np.stack(all_scores).mean(0)
        flat = [s for g in greedy for s in g]
        replay_env_seed = 777
        replay = record_replay(cls, best_q, replay_env_seed)
        results[name] = {
            "episodes": cfg["episodes"],
            "gamma": cfg["gamma"],
            "lr": cfg["lr"],
            "curve": bucket(mean_curve),
            "greedy_mean": float(np.mean(flat)),
            "greedy_max": int(max(flat)),
            "greedy_per_seed": [float(np.mean(g)) for g in greedy],
            "replay": replay,
        }
        print(f"{name}: greedy mean {np.mean(flat):.2f}  max {max(flat)}  "
              f"replay score {replay['score']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
