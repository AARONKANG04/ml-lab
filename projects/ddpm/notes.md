# Learning Notes on the DDPM Paper

> This paper presents progress in diffusion probabilistic models [ 53]. A diffusion probabilistic model
(which we will call a “diffusion model” for brevity) is a parameterized Markov chain trained using
variational inference to produce samples matching the data after finite time. Transitions of this chain
are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the
data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of
small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional
Gaussians too, allowing for a particularly simple neural network parameterization.

**Notes:**
- **Forward Process**: fixed Markov chain that gradually adds Gaussian noise to data over $T$ steps until it's pure noise
- **Reverse Process**: learned Markov chain that denoises step by step - this is parameterized by a neural network
- **Training via Variational Inference**: maximizing ELBO loss, same idea as VAEs but over a Markov chain
- **Neural Network simplification**: because the noise in the forward process is small and Gaussian, the reverse transitions are also approximately Gaussian, so the network just needs to predict Gaussian parameters at each step, which is a simple supervised problem 

<br>

> Diffusion models are straightforward to define and efficient to train, but to the best of our knowledge,
there has been no demonstration that they are capable of generating high quality samples. We
show that diffusion models actually are capable of generating high quality samples, sometimes
better than the published results on other types of generative models (Section 4). In addition, we
show that a certain parameterization of diffusion models reveals an equivalence with denoising
score matching over multiple noise levels during training and with annealed Langevin dynamics
during sampling (Section 3.2) [ 55, 61 ]. We obtained our best sample quality results using this
parameterization (Section 4.2), so we consider this equivalence to be one of our primary contributions.

**Notes:**
- Diffusion models were easy to train but nobody had shown that they could generate high quality images
- Predicting noise $\epsilon$ instead of $\mu$ directly is equivalent to learning how far any noisy image is from real data, which is a richer training signal than just predicting the next step. This is why it produces better samples, and connecting it to existing score matching literature gave it theoretical grounding 
- This parameterization also happens to give the best sample quality empirically, so the theory and practice align

<br>

> Despite their sample quality, our models do not have competitive log likelihoods compared to other
likelihood-based models (our models do, however, have log likelihoods better than the large estimates
annealed importance sampling has been reported to produce for energy based models and score
matching [11 , 55 ]). We find that the majority of our models’ lossless codelengths are consumed
to describe imperceptible image details (Section 4.3). We present a more refined analysis of this
phenomenon in the language of lossy compression, and we show that the sampling procedure of
diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit
ordering that vastly generalizes what is normally possible with autoregressive models

**Notes:**
- DDPM generates great-looking images but has worse log-likelihoods than other likelihood-based models (i.e. VAEs and flows). Good sample quality does not always imply high likelihood
- **Why?**: Most of the model's "bits" are spent encoding imperceptible details, tiny pixel-level noise that we can't see. The model is technically precise but that precision doesn't translate to perceptual quality
- You can think of the reverse process as progressive decoding - coarse structure first, fine-details later. This is similar to autoregressive models but over a much more flexible ordering than raw token-by-token, left-to-right
- **Takeaway**: log-likelihood is a flawed metric for generative models. A model can score poorly on it while still generating perceptually excellent samples

## 2. Background

> Diffusion models [53] are latent variable models of the form $p_\theta(\mathbf{x}_0) := \int p_\theta(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}$, where $\mathbf{x}_1, ..., \mathbf{x}_T$ are latents of the same dimensionality as the data $\mathbf{x}_0\sim q(\mathbf{x}_0)$. The joint distribution $p_\theta (\mathbf{x}_{0:T})$ is called the *reverse process*, and it is defined as a Markov chain with learned Gaussian transitions starting at $p(\mathbf{x}_T)=\mathcal{N}(\mathbf{x}_T;\mathbf{0}, \mathbf{I})$:
> $$p_\theta (\mathbf{x}_{0:T}):=p(\mathbf{x}_T)\prod_{t=1}^Tp_\theta(\mathbf{x}_{t-1}|x_t), \quad p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t):=\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

**Notes:**
- Diffusion models are just latent variable models. The latents $\mathbf{x}_1, ..., \mathbf{x}_T$ are just noisier versions of the data, same dimensionality as $\mathbf{x}_0$
- $p_\theta$ is a distribution over the data space, not a scalar. When you write $\log p_\theta(\mathbf{x}_0)$ you're evaluating the log probability density at a specific point, which gives a scalar, but $p_\theta$ itself integrates to 1 over all possible $\mathbf{x}_0$. For a well trained model, $p_\theta(\mathbf{x}_0)\approx 0$ everywhere except near the data manifold - the density is concentrated in a tiny region of the high-dimensional image space
- $p_\theta(\mathbf{x}_0)$ is the probability density of generating a specific clean image. Since there are infinitely many noise trajectories $\mathbf{x}_1, ..., \mathbf{x}_T$ that could lead to the same $\mathbf{x}_0$, you have to integrate over all of them - that's what $\int p_\theta(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}$ is doing. This is integral is completely intractable to compute directly, which is why DDPM optimizes the ELBO instead (same trick as VAEs)
- The full picture: the forward process $q$ gradually destroys a real image into pure noise. The reverse process $p_\theta$ learns to undo that, funneling probability density back towards the data manifold. Training maximizes $p_\theta(\mathbf{x}_0)$ over the training data, so the learned reverse transitions concentrate probability density where real images live. At inference you sample $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$ and run the reverse process. Because the model has learned to concentrate probability density on the data manifold, you're likely to land on a realistic image
- The reverse process $p_\theta(\mathbf{x}_{0:T})$ is the joint probability of an entire trajectory from pure noise back to a clean image. It factorizes as a Markov chain - each step only depends on the previous one - which gives the product $\prod_{t=1}^Tp_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$.
- Each reverse step is a Gaussian: the network predicts the mean $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ and covariance $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ conditioned on the current noisy image and timestep. The chain starts at pure noise $p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0}, \mathbf{I})$, and iteratively denoises toward $\mathbf{x}_0$

<br>

> What distinguishes diffusion models from other types of latent variable models is that the approximate posterior $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$, called the *forward process* or *diffusion process*, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_1, ..., \beta_T$:
> $$q(\mathbf{x}_{1:T}|\mathbf{x}_0):=\prod_{t=1}^Tq(\mathbf{x}_t|\mathbf{x}_{t-1}), \quad q(\mathbf{x}_t|\mathbf{x}_{t-1}):=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

**Notes:**
- In a VAE, the approximate posterior $q(\mathbf{x}_1|\mathbf{x}_0)$ is learned - an encoder network chooses how to map data to latents. In DDPM, the forward process $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ is fixed - no parameters, no learning. It's just a predefined noise schedule. This is what makes diffusion models unique
- The forward process is a Markov chain that gradually adds Gaussian noise over $T$ steps. Each step is: take the previous image, scale it down slightly by $\sqrt{1-\beta_t}$, and add Gaussian noise with variance $\beta_t$. After enough steps, $\mathbf{x}_T$ is pure Gaussian noise regardless of what $\mathbf{x}_0$ was
- The $\sqrt{1-\beta_t}$ scaling is important - without it, the signal would grow unboundedly as you keep adding noise. Scaling down preserves the overall variance and ensures the process converges to $\mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\beta_1, ..., \beta_T$ is the variance schedule - it controls how much noise is added at each step. Small $\beta_t$ means small gradual noising and large $\beta_t$ means aggressive noising. DDPM uses a linear schedule from $\beta_1=10^{-4}$ to $\beta_T=0.02$
- Since the forward process is fixed and has no parameters, it's called the approximate posterior rather than a learned one

<br>

> Training is performed by optimizing the usual variational bound on negative log likelihood:
> $$\mathbb{E}[-\log p_\theta(\mathbf{x}_0)]\leq \mathbb{E}_q\left[-\log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right]=\mathbb{E}_q\left[-\log p(\mathbf{x}_T)-\sum_{t\geq 1}\log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right]=: L$$
> The forward process variances $\beta_t$ can be learned by reparameterization or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of Gaussian conditionals in $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$, because both processes have the same functional form when $\beta_t$ is small. A notable property of the forward process is that it admits sampling $\mathbf{x}_t$ at an arbitrary timestep $t$ in closed form: using the notation $\alpha_t := 1-\beta_t$ and $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$, we have
> $$q(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}}\mathbf{x}_0, (1-\bar{\alpha})\mathbf{I})$$
> Efficient training is thereform possible by optimizing random terms of $L$ with stochastic gradient descent. Further improvements come from variance reduction by rewriting $L$ as:
> $$\mathbb{E}_q\left[\underbrace{D_\text{KL}(q(\mathbf{x}_T|\mathbf{x}_0)||p(\mathbf{x}_T))}_{L_T} + \sum_{t>1}\underbrace{D_\text{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)||p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}}\underbrace{-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0}\right]$$

**Notes:**
- The training objective is the standard ELBO from variational inference - the same idea as VAEs. Since $p_\theta(\mathbf{x}_0)$ is intractable (requires integrating over all trajectories), you instead minimize an upper bound on the negative log likelihood. The bound comes from Jensen's inequality applied to the log, and introduces the forward process $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ as the approximate posterior
- The ELBO decomposes into a sum over timesteps - each term compares the learned reverse step $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ against the forward step $q(\mathbf{x}_t|\mathbf{x}_{t-1})$. Intuitively, the model is being asked: "how well does your denoising distribution match the noising distribution in reverse?" Each term is a KL divergence in disguise
- The $\beta_t$ schedule can be learned or fixed. DDPM fixes them as constants, which simplifies training considerably
- The most important property introduced here is the closed-form sampling trick. Normally to get $\mathbf{x}_t$, you'd have to apply the forward process $t$ times sequentially. But because each step is Gaussian and Gaussians compose cleanly, you can collapse all $t$ steps into one: $$q(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}}\mathbf{x}_0, (1-\bar{\alpha})\mathbf{I})$$ where $\bar{\alpha}=\prod_{s=1}^t(1-\beta_s)$ is the cumulative product of signal retention across all steps. This means you can jump directly to any noise level in one shot, which makes training efficient. You don't need to simulate the full forward chain for every sample
- Intuitively, $\sqrt{\bar{\alpha}_t}$ controls how much of the original signal $\mathbf{x}_0$ remains at step $t$, and $(1-\bar{\alpha}_t)$ is how much variance the noise contributes. As $t\rightarrow T$, $\bar{\alpha}\rightarrow 0$and the image becomes pure noise

> Equation (5) uses KL divergence to directly compare $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ against forward process posteriors, which are tractable when conditioned on $\mathbf{x}_0$:
> $$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)=\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\tilde{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I}),$$ $$\text{where} \quad \boldsymbol{\tilde{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t \quad \text{and}\quad \tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$
> Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

**Notes:**
- The forward process posterior $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ is asking: given that we know both the noisy image at step $t$ *and* the original clean image $\mathbf{x}_0$, what's the distribution over the previous step $\mathbf{x}_{t-1}$? Conditioning on $\mathbf{x}_0$ is what makes this tractable. Without it, $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ requires marginalizing over all possible $\mathbf{x}_0$, which is intractable. With $\mathbf{x}_0$ known, it's just Bayes' rule applied to Gaussians, which has a closed form expression.
- The posterior mean $\boldsymbol{\tilde{\mu}}_t$ is a weighted interpolation between $\mathbf{x}_0$ and $\mathbf{x}_t$. Intuitively, the best guess for $\mathbf{x}_{t-1}$ is somewhere between the cleam image and the current noisy image, with the weighting determined by how much signal vs noise exists at each timestep. Early in the reverse process (large $t$, low $\bar{\alpha}_t$) it leans towards $\mathbf{x}_t$; late in the reverse process (small $t$, high $\bar{\alpha}_t$) it leans toward $\mathbf{x}_0$
- Each reverse step does two things: it moves closer to the clean image by stepping toward $\tilde{\boldsymbol{\mu}}_t$, then samples from the Gaussian around that mean rather jumping to it directly. This added noise is intentional - if you always moved exactly to the mean, the sampling process would be fully deterministic and every run from the same $\mathbf{x}_T$ would produce the identical output, collapsing the diversity of generated samples. The stochasticity from sampling around the mean is what allows the model to generate different images from different noise seeds
- The variance $\tilde{\beta}_t$ is fixed and has no learnable parameters - only the mean needs to be learned
- The key payoff is that since both $q(\mathbf{x}_{t-1}|\mathbf{x_t}, \mathbf{x}_0)$ and $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ are Gaussian, every KL divergence term in the ELBO has a closed form expression. The KL divergence between two Gaussians $\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$ is just: $$D_\text{KL}=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}$$ This is what "Rao-Blackwellized" means here. Instead of estimating the KL via high-variance Monte Carlo sampling, you compute it exactly in closed form, which gives much lower variance gradients and more stable training


## 3. Diffusion Models and Denoising Autoencoders

> Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances $\beta_t$ of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching that leads to a simplified, weighted variational bound objective for diffusion models. Ultimately, our model design is justified by simplicit and empirical results. Our descussion in dategorized by the terms of Eq. (5).

> **3.1  Forward process and $L_T$**
<br>
> We ignore the fact that the forward process variances $\beta_t$ are learnable by reparamaterization and instead fix them to constants. Thus, in our implementation, the approximate posterior $q$ has no learnable parameters, so $L_T$ is a constant during training and can be ignored. 

**Notes:**
- $L_T$ is the first KL term in the ELBO: $D_\text{KL}(q(\mathbf{x}_T|\mathbf{x}_0)||p(\mathbf{x}_T))$. It measures how different the fully-noised image distribution is from the prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$. Ideally these should be identical - if the forward process runs long enough, $\mathbf{x}_T$ should be indistinguishable from pure Gaussian noise regardless of $\mathbf{x}_0$
- By fixing $\beta_t$ as constants rather than learning them, $q$ has no learnable parameters at all. This means $L_T$ is just fixed number throughout training - it doesn't change, so it contributes nothing to the gradient and can simply be dropped from the loss. It's a constant offset on the ELBO that doesn't affect optimization
- This is a deliberate simplification that makes training cleaner. The tradeoff is that you lose some flexibility - a learned schedule could in principle better match $q(\mathbf{x}_T|\mathbf{x}_0)$ to $p(\mathbf{x}_T)$. But in practice, as long as $T$ is large enough and $\beta_T$ is large enough, the forward process destroys the signal completely and $L_T\approx 0$ anyway, so ignoring it has negligible effect

<br>

> **3.2 Reverse process and $L_{1:T-1}$**
<br>
> Now we discuss our choices in $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$ for $1<t\leq T$. First, we set $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)=\sigma_t^2\mathbf{I}$ to untrained time dependent constants. Experimentally, both $\sigma_t^2=\beta_t$ and $\sigma_t^2=\tilde{\beta_t}=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ had similar results. The first choice is optimal for $\mathbf{x}_0\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$, and the second is optimal for $\mathbf{x}_0$ deterministically set to one point. These are two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance. 

**Notes:**
- The reverse process has two learnable components: the mean $\boldsymbol{\mu}_\theta$ and the covariance $\boldsymbol{\Sigma}_\theta$. This section justifies fixing $\boldsymbol{\Sigma}_\theta=\sigma_t^2\mathbf{I}$ - a scalar times identity, with no learned parameters - and only learning then mean
- The two choices for $\sigma_t^2$ correspond to two extreme assumptions about the data:
    - $\sigma_t^2=\beta-t$ assumes maximum uncertainty - optimal when $\mathbf{x}_0$ is itself Gaussian, meaning the data has high entropy and many possible values of $\mathbf{x}_0$ are consistent with $\mathbf{x}_t$
    - $\sigma_t^2=\tilde{\beta_t}=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ assumes minimum uncertainty - optimal when $\mathbf{x}_0$ is a single fixed point, meaning there's essentially no ambiguity about which clean image produced $\mathbf{x}_t$
- Real data lives somewhere between these extremes, and the paper finds both choices give similar results in practice, so the simpler $\sigma_t^2=\beta_t$ is used. The key insight is that the variance matters much less than the mean for sample quality, which is why it's safe to fix it and spend all the model capacity on learning $\boldsymbol{\mu}_\theta$

<br>

> Second, to represent the mean $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$, we propose a specific parameterization motivated by the following analysis of $L_t$. With $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I}})$, we can write: $$L_{t-1}=\mathbb{E}_q\left[\frac{1}{2\sigma_t^2}||\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)-\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)||^2\right]+C$$ where $C$ is a constant that does not depend on $\theta$. So, we see that the most straightforward parameterization of $\boldsymbol{\mu}_\theta$ is a model that predicts $\tilde{\boldsymbol{\mu}}_t$, the forward process posterior mean. However, we can expand Eq. (8) further by reparameterizing Eq. (4) as $\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon})=\sqrt{\bar{\alpha}}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ for $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$ and applying the forward process posterior formula: $$L_{t-1}-C=\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{1}{2\sigma_t^2}\left|\left|\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}), \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon})-\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})\right)-\boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}), t)\right|\right|^2\right]$$ $$\qquad=\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{1}{2\sigma_t^2}\left|\left|\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon})-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)-\boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}), t)\right|\right|^2\right]$$

**Notes:**
- Since both $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ and $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ are Gaussians with the same fixed variance $\sigma_t^2\mathbf{I}$, the KL divergence between them collapses to just a weighted squared distance between their means - that''s where the $||\tilde{\boldsymbol{\mu}}_t-\boldsymbol{\mu}_\theta||^2$ term comes from. So minimizing $L_{t-1}$ is equivalent to making the predicted mean match the true posterior mean
- The naive approach is to predict $\tilde{\mu}_t$ directly. But the paper instead substitutes in the reparameterization $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$, solving for $\mathbf{x}_0$: $$\mathbf{x}_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})$$ Plugging this into the formula for $\tilde{\boldsymbol{\mu}}_t$ and simplifying gives the second equation - the target mean is now expressed purely in terms of $\mathbf{x}_t$ and $\boldsymbol{\epsilon}$, suggesting the parameterization: $$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$ The network predicts $\boldsymbol{\epsilon}$ and the mean is recovered analytically for free. 
- **Why predicting $\boldsymbol{\epsilon}$ is better than predicting $\tilde{\boldsymbol{\mu}}_t$** directly: $\tilde{\boldsymbol{\mu}}_t$ is an unstandardized target whose scale and magnitude are modulated by $\sqrt{\bar{\alpha}_t}$ and $\sqrt{1-\bar{\alpha}_t}$, which sweep across a wide range as $t$ goes from $1$ to $T$. At large $t$, the target mean is dominatd by $\mathbf{x}_t$; at small $t$, it's dominated by $\mathbf{x}_0$ - a completely different optimization target at every timestep. The network would have to implicitly learn this rescaling on top of the actual denoising task. 
- Crucially, the conditioning on $t$ is not optional - the network must know which timestep it's at because the "vector field" it needs to predict points in fundamentally different directions depending on $t$. Close to $T$, the noisy image $\mathbf{x}_t$ is nearly pure Gaussian noise and the predicted $\boldsymbol{\epsilon}$ should point the reverse process toward the general mean of the data manifold - coarse structure like the overall shape and layout of an image. Close to $t=0$, the image is nearly clean and the predicted $\boldsymbol{\epsilon}$ captures fine-grained details like textures and sharp edges. The same network handles both regimes, with $t$ as an input telling it which regime it's operating in. This is why diffusion models use a timestep embedding (typically a sinusoidal or learned embedding, analogous to positional encodings in transformers) - without it, the network has no way to know whether it should be recovering coarse structure or refining fine details. 
- By contrast, $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$ is always a unit Gaussian sample - the same distribution, the same scale, every timestep. The network is always solving the same type of problem: "what noise was added to this image?" The timestep-dependent rescaling is handled analytically by the fixed formula above, not by learned weights. It's a standardized target versus an unstandardized one - and standardized targets are almost always easier to optimize. 

> $$\begin{array}{l}
\textbf{Algorithm 1 } \text{Training} \\
\hline
1:\ \textbf{repeat} \\
2:\ \quad \mathbf{x}_0 \sim q(\mathbf{x}_0) \\
3:\ \quad t \sim \text{Uniform}(\{1, \ldots, T\}) \\
4:\ \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
5:\ \quad \text{Take gradient descent step on} \\
\qquad\qquad \nabla_\theta \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\, t\right) \right\|^2 \\
6:\ \textbf{until } \text{converged} \\
\hline
\end{array}$$



> Equation (10) reveals that $\boldsymbol{\mu}_\theta$ must predict $\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)$ given $\mathbf{x}_t$. Since $\mathbf{x}_t$ is available as input to the model, we may choose the parameterization $$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)=\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t))\right)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$ where $\boldsymbol{\epsilon}_\theta$ is a function approximator intended to predict $\boldsymbol{\epsilon}$ from $\mathbf{x}_t$. To sample $\mathbf{x}_{t-1}\sim p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ is to compute $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}$ where $\mathbf{z}\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$. The complete sampling procedure, Algorithm 2, resembles Langevin dynamics with $\boldsymbol{\epsilon}_\theta$ as a learned gradient of the data density. Furthermore, with the parameterization 911), Eq. (10) simplifies to: $$\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1-\bar{\alpha}_t)}\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\, t)\right\|^2\right]$$ which resembles denoising score matching over multiple noise scales indexed by $t$. As Eq. (12) is equal to (one term of) the variational bound for the Langevin-like reverse process (11), we see that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics. 

**Notes:**
- To generate $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$, you run: $$\mathbf{x}_{t-1}=\underbrace{\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)}_{\text{predicted mean }\boldsymbol{\mu}_\theta}+\underbrace{\sigma_t\mathbf{z}}_{\text{stochastic noise}}$$ The first term is the predicted mean - you take the current noisy image, subtract the predicted noise (scaled appropriately), and rescale. The second term $\sigma_t\mathbf{z}$ is the intentional stochasticity that keeps sampling from being deterministic.
- Plugging the $\boldsymbol{\epsilon}_\theta$ parameterization back into $L_{t-1}$ simplifies it to just a weighted MSE between the true noise and the predicted noise: $$\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\,\bigg\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\, t\right)\bigg\|^2\right]$$ The weighted coefficient $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}$ is a timestep-dependent scalar - in the simplified objective, the weights are dropped entirely and just uses unweighted MSE, which works better empirically.
- The connection to Langevin dynamics and denoising score matching is the deep theoretical result in this paper. Langevin dynamics is a sampling algorithm that iteratively moves toward high-density regions of a distribution by following the score $\nabla_x\log p(x)$ plus injected noise. The sampling procedure here does exactly this - $\boldsymbol{\epsilon}_\theta$ is implicitly estimating the score of the data distribution (since $\nabla_{x_t}\log p(x_t)\propto -\boldsymbol{\epsilon}$), and $\sigma_t\mathbf{z}$ is the injected noise. The fact that the ELBO objective (a principled variation inference bound) is what unifies the two research threads - DDPM wasn't just a new ad-hoc method

<br>

> $$\begin{array}{l}
\textbf{Algorithm 2 } \text{Sampling}
\\
\hline
1:\ \mathbf{x}_T\sim\mathcal{N}(\mathbf{0}, \mathbf{I}) \\
2:\ \textbf{for } t = T,...,1 \textbf{ do} \\
3: \ \quad \mathbf{z}\sim\mathcal{N}(\mathbf{0}, \mathbf{I}) \text{ if } t > 0, \text{ else } \mathbf{z}=\mathbf{0} \\
4:\ \quad \mathbf{x}_{t-1}=\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)+\sigma_t\mathbf{z} \\
5:\ \textbf{end for} \\
6:\ \textbf{return } \mathbf{x}_0 \\
\hline
\end{array}$$ 


> To summarize, we can train the reverse process mean function approximator $\boldsymbol{\mu}_\theta$ to predict $\tilde{\boldsymbol{\mu}}_t$, or by modifying its parameterization, we can train it to predict $\boldsymbol{\epsilon}$. (There is also the possibly of predicting $\mathbf{x}_0$, but we found this to lead to worse sample quality early in our experiments.) We have shown that the $\boldsymbol{\epsilon}$-prediction parameterization both resembles Langevin dynamics and simplifies the diffusion model's variational bound to an objective that resembles denoising score matching. Nonetheless, it is just another parameterization of $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$, so we verify its effectiveness in Section 4 in an ablation where we compare predicting $\boldsymbol{\epsilon}$ against predicting $\tilde{\boldsymbol{\mu}}_t$