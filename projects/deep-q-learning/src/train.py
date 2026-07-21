import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dqn import QNet, ReplayBuffer, device, td_loss
from games import Breakout, Snake

GAMES = {"snake": Snake, "breakout": Breakout}


def stack(prev, cur):
    return np.concatenate([prev, cur], axis=0)


def act_greedy(online, s, dev):
    x = torch.as_tensor(s[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        return int(online(x).argmax())


def greedy_episode(cls, online, dev, seed, record=False, max_steps=3000,
                   max_frames=400):
    env = cls(seed=seed)
    cur = env.reset()
    prev = cur
    frames = [env.frame()] if record else None
    for _ in range(max_steps):
        a = act_greedy(online, stack(prev, cur), dev)
        prev = cur
        cur, _, done = env.step(a)
        if record and len(frames) < max_frames:
            frames.append(env.frame())
        if done:
            break
    return env.score, frames


def evaluate(cls, online, dev, n=10, base_seed=50000):
    return [greedy_episode(cls, online, dev, base_seed + i)[0] for i in range(n)]


def train(game, steps, seed, out_dir, gamma=0.99, lr=2.5e-4, batch_size=64,
          buffer_cap=100000, warmup=5000, train_every=2, target_every=2000,
          eps_hi=1.0, eps_lo=0.05, eval_every=20000):
    cls = GAMES[game]
    dev = device()
    env = cls(seed=seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    in_ch = 2 * cls.obs_channels
    online = QNet(in_ch, cls.H, cls.W, cls.n_actions).to(dev)
    target = QNet(in_ch, cls.H, cls.W, cls.n_actions).to(dev)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_cap, (in_ch, cls.H, cls.W), seed=seed)
    eps_decay_steps = int(0.3 * steps)

    cur = env.reset()
    prev = cur
    recent = deque(maxlen=100)
    history = {"train": [], "eval": []}
    t0 = time.time()

    for step in range(1, steps + 1):
        eps = max(eps_lo, eps_hi - (eps_hi - eps_lo) * step / eps_decay_steps)
        s = stack(prev, cur)
        if rng.random() < eps:
            a = int(rng.integers(cls.n_actions))
        else:
            a = act_greedy(online, s, dev)
        nxt, r, done = env.step(a)
        s2 = stack(cur, nxt)
        buf.push(s, a, r, s2, done)
        if done:
            recent.append(env.score)
            cur = env.reset()
            prev = cur
        else:
            prev = cur
            cur = nxt

        if step >= warmup and step % train_every == 0:
            loss = td_loss(online, target, buf.sample(batch_size, dev), gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if step % target_every == 0:
            target.load_state_dict(online.state_dict())

        if step % eval_every == 0:
            scores = evaluate(cls, online, dev)
            history["eval"].append({"step": step, "scores": scores})
            history["train"].append({"step": step,
                                     "score": float(np.mean(recent or [0]))})
            rate = step / (time.time() - t0)
            print(f"{game} step {step:7d}  eps {eps:.2f}  "
                  f"train100 {np.mean(recent or [0]):6.2f}  "
                  f"eval {np.mean(scores):6.2f} (max {max(scores)})  "
                  f"{rate:.0f} steps/s", flush=True)

    final = evaluate(cls, online, dev, n=30, base_seed=90000)
    best_seed, best_score = 0, -1
    for i in range(10):
        sc, _ = greedy_episode(cls, online, dev, 70000 + i)
        if sc > best_score:
            best_score, best_seed = sc, 70000 + i
    replay_score, frames = greedy_episode(cls, online, dev, best_seed, record=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{game}.json").write_text(json.dumps({
        "game": game, "steps": steps, "seed": seed,
        "final_scores": final,
        "final_mean": float(np.mean(final)), "final_max": int(max(final)),
        "history": history,
        "replay": {"frames": frames, "score": replay_score},
    }))
    torch.save(online.state_dict(), out_dir / f"{game}.pt")
    print(f"{game} done: final {np.mean(final):.2f} (max {max(final)}), "
          f"replay {replay_score}, saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=list(GAMES), required=True)
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    train(args.game, args.steps, args.seed, args.out)


if __name__ == "__main__":
    main()
