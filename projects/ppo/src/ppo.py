import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        bm = x.mean(axis=0)
        bv = x.var(axis=0)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)

    def state(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(),
                "count": self.count}

    def load(self, s):
        self.mean = np.array(s["mean"])
        self.var = np.array(s["var"])
        self.count = s["count"]


def layer(in_dim, out_dim, gain):
    lin = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.constant_(lin.bias, 0.0)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            layer(obs_dim, hidden, np.sqrt(2)), nn.Tanh(),
            layer(hidden, hidden, np.sqrt(2)), nn.Tanh(),
            layer(hidden, act_dim, 0.01),
        )
        self.critic = nn.Sequential(
            layer(obs_dim, hidden, np.sqrt(2)), nn.Tanh(),
            layer(hidden, hidden, np.sqrt(2)), nn.Tanh(),
            layer(hidden, 1, 1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


def gae(rewards, values, dones, last_value, gamma, lam):
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last = delta + gamma * lam * not_done * last
        adv[t] = last
    return adv, adv + values


def ppo_update(net, opt, batch, clip=0.2, epochs=10, minibatches=32,
               vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5):
    obs, actions, old_logp, advantages, returns = batch
    n = obs.shape[0]
    idx = np.arange(n)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "kl": 0.0, "clipfrac": 0.0}
    count = 0
    for _ in range(epochs):
        np.random.shuffle(idx)
        for mb in np.array_split(idx, minibatches):
            mo, ma = obs[mb], actions[mb]
            adv = advantages[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            d = net.dist(mo)
            logp = d.log_prob(ma).sum(-1)
            ratio = (logp - old_logp[mb]).exp()
            pi_loss = torch.max(-adv * ratio,
                                -adv * ratio.clamp(1 - clip, 1 + clip)).mean()
            v_loss = 0.5 * (net.value(mo) - returns[mb]).pow(2).mean()
            entropy = d.entropy().sum(-1).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["kl"] += (old_logp[mb] - logp).mean().item()
                stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
            count += 1
    return {k: v / count for k, v in stats.items()}
