import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ppo import ActorCritic, RunningMeanStd, gae, ppo_update


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


class RewardScaler:
    def __init__(self, num_envs, gamma):
        self.returns = np.zeros(num_envs)
        self.gamma = gamma
        self.rms = RunningMeanStd(())

    def scale(self, rewards, dones):
        self.returns = self.returns * self.gamma * (1.0 - dones) + rewards
        self.rms.update(self.returns)
        return np.clip(rewards / np.sqrt(self.rms.var + 1e-8), -10.0, 10.0)


def evaluate(env_id, net, obs_rms, dev, episodes=5, record=False):
    env = gym.make(env_id)
    returns, lengths, best = [], [], None
    for ep in range(episodes):
        obs, _ = env.reset(seed=123400 + ep)
        traj = []
        total, steps, done = 0.0, 0, False
        while not done:
            if record:
                d = env.unwrapped.data
                traj.append((d.qpos.copy(), d.qvel.copy()))
            x = torch.as_tensor(obs_rms.normalize(obs[None]), dtype=torch.float32,
                                device=dev)
            with torch.no_grad():
                a = net.actor(x)[0].cpu().numpy()
            obs, r, term, trunc, _ = env.step(np.clip(a, env.action_space.low,
                                                      env.action_space.high))
            total += r
            steps += 1
            done = term or trunc
        returns.append(total)
        lengths.append(steps)
        if record and (best is None or total > best[0]):
            best = (total, traj)
    env.close()
    out = {"mean": float(np.mean(returns)), "max": float(np.max(returns)),
           "min": float(np.min(returns)), "len": float(np.mean(lengths))}
    if record and best is not None:
        out["trajectory"] = [[round(v, 4) for v in np.concatenate(p)]
                             for p in best[1]]
    return out


def train(env_id, total_steps, num_envs, rollout, seed, out_dir,
          lr=3e-4, gamma=0.99, lam=0.95, eval_every=40):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gym.vector.AsyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    net = ActorCritic(obs_dim, act_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    obs_rms = RunningMeanStd((obs_dim,))
    rscale = RewardScaler(num_envs, gamma)

    updates = total_steps // (num_envs * rollout)
    obs, _ = envs.reset(seed=seed)
    ep_ret = np.zeros(num_envs)
    recent = []
    history = {"train": [], "eval": []}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for update in range(1, updates + 1):
        frac = 1.0 - (update - 1) / updates
        for g in opt.param_groups:
            g["lr"] = lr * frac

        buf_obs = np.zeros((rollout, num_envs, obs_dim), dtype=np.float32)
        buf_act = torch.zeros((rollout, num_envs, act_dim), device=dev)
        buf_logp = torch.zeros((rollout, num_envs), device=dev)
        buf_val = torch.zeros((rollout, num_envs), device=dev)
        buf_rew = torch.zeros((rollout, num_envs), device=dev)
        buf_done = torch.zeros((rollout, num_envs), device=dev)

        for t in range(rollout):
            obs_rms.update(obs)
            norm = obs_rms.normalize(obs).astype(np.float32)
            buf_obs[t] = norm
            x = torch.as_tensor(norm, device=dev)
            with torch.no_grad():
                a, logp, v = net.act(x)
            buf_act[t], buf_logp[t], buf_val[t] = a, logp, v
            obs, r, term, trunc, _ = envs.step(a.cpu().numpy())
            done = np.logical_or(term, trunc).astype(np.float64)
            ep_ret += r
            for i in np.flatnonzero(done):
                recent.append(ep_ret[i])
                ep_ret[i] = 0.0
            recent = recent[-100:]
            buf_rew[t] = torch.as_tensor(rscale.scale(r, done), device=dev,
                                         dtype=torch.float32)
            buf_done[t] = torch.as_tensor(done, device=dev, dtype=torch.float32)

        with torch.no_grad():
            last_v = net.value(torch.as_tensor(
                obs_rms.normalize(obs).astype(np.float32), device=dev))
        adv, ret = gae(buf_rew, buf_val, buf_done, last_v, gamma, lam)

        flat = lambda x: x.reshape(-1, *x.shape[2:])
        batch = (torch.as_tensor(buf_obs.reshape(-1, obs_dim), device=dev),
                 flat(buf_act), flat(buf_logp), flat(adv), flat(ret))
        stats = ppo_update(net, opt, batch)

        steps_done = update * num_envs * rollout
        history["train"].append({"step": steps_done,
                                 "return": float(np.mean(recent or [0])),
                                 **{k: round(v, 5) for k, v in stats.items()}})

        if update % eval_every == 0 or update == updates:
            ev = evaluate(env_id, net, obs_rms, dev)
            history["eval"].append({"step": steps_done, **ev})
            sps = steps_done / (time.time() - t0)
            print(f"update {update:4d}/{updates}  step {steps_done:9d}  "
                  f"train100 {np.mean(recent or [0]):8.1f}  "
                  f"eval {ev['mean']:8.1f} (len {ev['len']:.0f})  "
                  f"kl {stats['kl']:.4f}  {sps:.0f} steps/s", flush=True)
            torch.save({"net": net.state_dict(), "obs_rms": obs_rms.state()},
                       out_dir / "checkpoint.pt")
            (out_dir / "results.json").write_text(json.dumps(
                {"env": env_id, "total_steps": total_steps, "seed": seed,
                 "num_envs": num_envs, "rollout": rollout, "history": history}))

    final = evaluate(env_id, net, obs_rms, dev, episodes=10, record=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"env": env_id, "total_steps": total_steps, "seed": seed,
         "num_envs": num_envs, "rollout": rollout, "history": history,
         "final": final}))
    print(f"final eval: mean {final['mean']:.1f}  max {final['max']:.1f}  "
          f"len {final['len']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Humanoid-v5")
    ap.add_argument("--total-steps", type=int, default=20000000)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    train(args.env, args.total_steps, args.num_envs, args.rollout, args.seed,
          args.out)


if __name__ == "__main__":
    main()
