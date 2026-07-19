# DDIM

Reimplementation of [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020).

An accelerated sampling method for diffusion models that generalizes DDPM to non-Markovian processes, enabling high-quality sample generation in far fewer steps.

## Algorithm

DDIM uses the same trained noise predictor $\epsilon_\theta$ as DDPM but replaces the stochastic reverse process with a deterministic (or partially stochastic) update rule.

### Generalized Forward Process

DDIM defines a family of non-Markovian forward processes indexed by $\sigma$ that all share the same marginals as DDPM:

$$q_\sigma(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}},\; \sigma_t^2 \mathbf{I}\right)$$

When $\sigma_t = \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t}$, this recovers DDPM. When $\sigma_t = 0$, the process becomes fully deterministic.

### Sampling Update

Given a noise prediction $\epsilon_\theta(\mathbf{x}_t, t)$, first predict $\mathbf{x}_0$:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

Then compute the next sample:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{\mathbf{x}}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(\mathbf{x}_t, t) + \sigma_t\, \mathbf{z}$$

where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ if $\sigma_t > 0$, else $\mathbf{z} = \mathbf{0}$.

### Accelerated Sampling

The key insight is that because the forward marginals $q(\mathbf{x}_t | \mathbf{x}_0)$ only depend on $\bar{\alpha}_t$, the reverse process can skip timesteps. Instead of iterating through all $T$ steps, DDIM uses a subsequence $\tau = [\tau_1, \tau_2, ..., \tau_S]$ where $S \ll T$, enabling generation in as few as 10 to 50 steps with minimal quality loss.

## Implementation Notes
- Uses the same pretrained $\epsilon_\theta$ from DDPM, no retraining needed
- Setting $\sigma_t = 0$ (deterministic DDIM) makes sampling fully deterministic: the same $\mathbf{x}_T$ always produces the same $\mathbf{x}_0$
- The $\eta$ parameter controls stochasticity: $\sigma_t = \eta \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t}$. $\eta = 0$ is deterministic DDIM, $\eta = 1$ recovers DDPM

## Results