# Learning Notes on the DDIM Paper

### Abstract
Denoising diffusion probabilistic models (DDPMs) have achieved high qual-
ity image generation without adversarial training, yet they require simulating a
Markov chain for many steps in order to produce a sample. To accelerate sam-
pling, we present denoising diffusion implicit models (DDIMs), a more efficient
class of iterative implicit probabilistic models with the same training procedure as
DDPMs. In DDPMs, the generative process is defined as the reverse of a particular
Markovian diffusion process. We generalize DDPMs via a class of non-Markovian
diffusion processes that lead to the same training objective. These non-Markovian
processes can correspond to generative processes that are deterministic, giving rise
to implicit models that produce high quality samples much faster. We empirically
demonstrate that DDIMs can produce high quality samples 10× to 50× faster in
terms of wall-clock time compared to DDPMs, allow us to trade off computation
for sample quality, perform semantically meaningful image interpolation directly
in the latent space, and reconstruct observations with very low error.

**Notes:**
- DDPMs achieve high image quality but require lots of compute. Sampling requires simulating the full Markov chain for $T\approx 1000$ steps, each requiring a full forward pass through the network. Unlike GANs which generate in a single pass, this is inherently sequential and slow.
- DDIMs achieve similar image quality with far fewer steps. They can be 10-50x faster using the exact same trained model as DDPM. No retraining needed as the key insight is that slow sampling in DDPM is a property of the sampling procedure rather than of the model itself
- The core idea is that DDPM's generative process is defined as the reverse of a Markovian diffusion process. Each step depends only on the previous step. DDIM generalizes this by finding a family of non-Markovian forward processes that use the same training objective. Since the training objective is identical, a model trained with DDPM can be sampled using DDIM's generative process without any modification


<br>

### 1 - Introduction

Deep generative models have demonstrated the ability to produce high quality samples in many
domains (Karras et al., 2020; van den Oord et al., 2016a). In terms of image generation, genera-
tive adversarial networks (GANs, Goodfellow et al. (2014)) currently exhibits higher sample quality
than likelihood-based methods such as variational autoencoders (Kingma & Welling, 2013), autore-
gressive models (van den Oord et al., 2016b) and normalizing flows (Rezende & Mohamed, 2015;
Dinh et al., 2016). However, GANs require very specific choices in optimization and architectures
in order to stabilize training (Arjovsky et al., 2017; Gulrajani et al., 2017; Karras et al., 2018; Brock
et al., 2018), and could fail to cover modes of the data distribution (Zhao et al., 2018).
Recent works on iterative generative models (Bengio et al., 2014), such as denoising diffusion prob-
abilistic models (DDPM, Ho et al. (2020)) and noise conditional score networks (NCSN, Song &
Ermon (2019)) have demonstrated the ability to produce samples comparable to that of GANs, with-
out having to perform adversarial training. To achieve this, many denoising autoencoding models
are trained to denoise samples corrupted by various levels of Gaussian noise. Samples are then
produced by a Markov chain which, starting from white noise, progressively denoises it into an im-
age. This generative Markov Chain process is either based on Langevin dynamics (Song & Ermon,
2019) or obtained by reversing a forward diffusion process that progressively turns an image into
noise (Sohl-Dickstein et al., 2015).
A critical drawback of these models is that they require many iterations to produce a high quality
sample. For DDPMs, this is because that the generative process (from noise to data) approximates
the reverse of the forward diffusion process (from data to noise), which could have thousands of
steps; iterating over all the steps is required to produce a single sample, which is much slower
compared to GANs, which only needs one pass through a network. For example, it takes around 20
hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from
a GAN on a Nvidia 2080 Ti GPU. This becomes more problematic for larger images as sampling
50k images of size 256 × 256 could take nearly 1000 hours on the same GPU.
To close this efficiency gap between DDPMs and GANs, we present denoising diffusion implicit
models (DDIMs). DDIMs are implicit probabilistic models (Mohamed & Lakshminarayanan, 2016)
and are closely related to DDPMs, in the sense that they are trained with the same objective function.


**Notes:**
- GANS had the best sample quality but were unstable to train and prone to mode collapse
- DDPM matched GAN quality without adversarial training, but requires $T\approx 1000$ sequential network passes to generate one sample
- Concretely on a NVIDIA 2080ti GPU, it takes 20 hours for 50k images at $32\times 32$ and nearly 1000 hours at $256\times 256$
- However, this slowness is a property of the sampling procedure, not the model
- The DDPM training objective only requires the marginals $q(\mathbf{x}_t|\mathbf{x}_0)$ to be correct. It doesn't actually require the forward process to be Markovian
- So finding a non-Markovian forward process with the same marginals induces the same $\boldsymbol{\epsilon}$-prediction loss
- A DDPM trained model can then be sampled with a faster non-Markovian procedure without retraining

<br>

In Section 3, we generalize the forward diffusion process used by DDPMs, which is Markovian,
to non-Markovian ones, for which we are still able to design suitable reverse generative Markov
chains. We show that the resulting variational training objectives have a shared surrogate objective,
which is exactly the objective used to train DDPM. Therefore, we can freely choose from a large
family of generative models using the same neural network simply by choosing a different, non-
Markovian diffusion process (Section 4.1) and the corresponding reverse generative Markov Chain.
In particular, we are able to use non-Markovian diffusion processes which lead to ”short” generative
Markov chains (Section 4.2) that can be simulated in a small number of steps. This can massively
increase sample efficiency only at a minor cost in sample quality.
In Section 5, we demonstrate several empirical benefits of DDIMs over DDPMs. First, DDIMs have
superior sample generation quality compared to DDPMs, when we accelerate sampling by 10× to
100× using our proposed method. Second, DDIM samples have the following “consistency” prop-
erty, which does not hold for DDPMs: if we start with the same initial latent variable and generate
several samples with Markov chains of various lengths, these samples would have similar high-level
features. Third, because of “consistency” in DDIMs, we can perform semantically meaningful image
interpolation by manipulating the initial latent variable in DDIMs, unlike DDPMs which interpolates
near the image space due to the stochastic generative process.

**Notes:**
- DDPM's forward process is one specific Markovian diffusion, but there's actually a whole family of non-Markovian forward processes that produce the same marginals $q(\mathbf{x}_t|\mathbf{x}_0)$ and therefore the same training objective
- Since the training objective is identical across this family, any model trained with DDPM implicity works for all of them. You get a free choice of generative process at inference time
- Within this family, you can choose processes whose reverse chains are much shorter, skipping steps entirely, giving 10-100x speedup at a minor quality cost
- Consistency property: In DDPM, the stochastic $\sigma_t\mathbf{z}$ terms added per step means the same $\mathbf{x}_T$ produces different samples every run. In DDIM (deterministic, $\sigma_t=0$), the same $\mathbf{x}_T$ always produces the same $\mathbf{x}_0$. This makes $\mathbf{x}_T$ a true latent code for the image
- A consequence of consistency is that you can interpolate between two images by interpolating their latent codes $\mathbf{x}_T^{(1)}$ and $\mathbf{x}_T^{(2)}$, and because the mapping is deterministic and smooth, the interpolation is semantically meaningful (blending high-level features, not just pixels). DDPM can't do this because its stochastic sampling breaks the latent-to-image correspondence



### 2 - Background

Given samples from a data distribution $q(\mathbf{x}_0)$, we are interested in learning a model distribution $p_\theta(\mathbf{x}_0)$ that approximates $q(\mathbf{x}_0)$ and is easy to sample from. Denoising diffusion probability models are latent variable models of the form $$p_\theta(\mathbf{x}_0)=\int p_\theta(\mathbf{x}_{0:T})d\mathbf{x}_{1:T},\quad\text{where}\quad p_\theta(\mathbf{x}_{0:T}):=p_\theta(\mathbf{x}_T)\prod_{t=1}^Tp_\theta^{(t)}(\mathbf{x}_{t-1}|\mathbf{x}_t)$$ where $\mathbf{x}_1, ..., \mathbf{x}_T$ are latent variables in the same sample space as $\mathbf{x}_0$ (denoted as $\mathcal{X}$). The parameters $\theta$ are learned to fit the data distribution $q(\mathbf{x}_0)$ by maximizing a variational lower bound: $$\max_\theta\mathbb{E}_{q(\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0)]\leq \max_\theta\mathbb{E}_{q(\mathbf{x}_0, ..., \mathbf{x}_T)}[\log p_\theta(\mathbf{x}_{0:T})-\log q(\mathbf{x}_{1:T}|\mathbf{x}_0)]$$ where $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ is some inference distribution over the latent variables. Unlike typical latent variable models (such as the variational autoencoder), DDPMs are learned with a fixed (rather than trainable) inference procedure $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$, and latent variables are relatively high dimensional. For example, Ho et al. (2020) considered the following Markov chain with Gaussian transitions parameterized by a decreasing sequence $\alpha_{1:T}\in (0,1]^T$: $$q(\mathbf{x}_{1:T}|\mathbf{x}_0):=\prod_{t=1}^Tq(\mathbf{x}_t|\mathbf{x}_{t-1}),\text{ where }q(\mathbf{x}_t|\mathbf{x}_{t-1}):=\mathcal{N}\left(\sqrt{\frac{\alpha_t}{\alpha_{t-1}}}\mathbf{x}_{t-1}, \left(1-\frac{\alpha_t}{\alpha_{t-1}}\right)\mathbf{I}\right)$$ where the covariance matrix is ensured to have positive terms on its diagonal. This is called the forward process due to the autoregressive nature of the sampling procedure (from $\mathbf{x}_0$ to $\mathbf{x}_T$). We call the latent variable model $p_\theta(\mathbf{x}_{0:T})$, which is a Markov chain that samples from $\mathbf{x}_T$ to $\mathbf{x}_0$, the *generative process*, since it approximates the intractable *reverse process* $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$. Intuitively, the forward process progressively adds noise to the observation $\mathbf{x}_0$, whereas the generative process progressively denoises a noisy observation. 

**Notes:**
- The fundamental goal is to learn a model distribution $p_\theta(\mathbf{x}_0)$ that approximates the true data distribution $q(\mathbf{x}_0)$. Since we can't directly maximize $\log p_\theta(\mathbf{x}_0)$ (it requires marginalizing over all trajectories, which is intractable), we introduce latent variables $\mathbf{x}_1, ..., \mathbf{x}_T$, progressively noisier versions of $\mathbf{x}_0$, and instead optimize the ELBO variational lower bound.
- One subtle notational difference from DDPM paper is that DDIM uses $\alpha_t$ to denote what DDPM called $\bar{\alpha}_t$, which is the cumulative signal retention up to timestep $t$, not the per-step factor. The per-step transition uses $\sqrt{\alpha_t / \alpha_{t-1}}$ as the signal scaling, which recovers the per-step factor by dividing consecutive cumulative products. 
- The important distinction is between the generative process $p_\theta(\mathbf{x}_{0:T})$ and the true reverse process $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$. The true reverse, "given this noisy image, what did it look like one step earlier?", is intracable because computing it requires knowing $p(\mathbf{x}_t)$, the marginal over all possible images at timestep $t$. The generative process is a learned approximation to this intractable reverse. The ELBO is used precisely because you can't optimize the true reverse directly. 


<br>

A special property of the forward process is that $$q(\mathbf{x}_t|\mathbf{x}_0):=\int q(\mathbf{x}_{1:t}|\mathbf{x}_0)d\mathbf{x}_{1:(t-1)}=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\mathbf{x}_0, (1-\alpha_t)\mathbf{I})$$ so we can express $\mathbf{x}_t$ as a linear combination of $\mathbf{x}_0$ and a noise variable $\epsilon$: $$\mathbf{x}_t=\sqrt{\alpha_t}\mathbf{x}_0+\sqrt{1-\alpha_t}\epsilon,\quad\text{where}\quad \epsilon\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$$ When we set $\alpha_T$ sufficiently close to $0$, $q(\mathbf{x}_T|\mathbf{x}_0)$ converges to a standard Gaussian for all $\mathbf{x}_0$, so it is natural to set $p_\theta(\mathbf{x}_T):=\mathcal{N}(\mathbf{0}, \mathbf{I})$. If all the conditionals are modeled as Gaussians with trainable mean functions and fixed variances, the objective can be simplified to: $$L_\gamma(\epsilon_\theta):=\sum_{t=1}^T\gamma_t\mathbb{E}_{\mathbf{x}_0\sim q(\mathbf{x}_0),\epsilon_t\sim\mathcal{N}(\mathbf{0}, \mathbf{I})}\left[\left\| \epsilon_\theta^{(t)}(\sqrt{\alpha_t}\mathbf{x}_0+\sqrt{1-\alpha_t}\epsilon_t)-\epsilon_t \right\|_2^2 \right]$$ where $\epsilon_\theta := \{\epsilon_\theta^{(t)}\}_{t=1}^T$ is a set of $T$ functions, each $\epsilon_\theta^{(t)}:\mathcal{X}\rightarrow\mathcal{X}$ (indexed by $t$) is a function with trainable parameters $\theta^{(t)}$, and $\gamma := [\gamma_1, ..., \gamma_T]$ is a vector of positive coefficients in the objective that depends on $\alpha_{1:T}$. In Ho et al. (2020), the objective with $\gamma=1$ is optimized instead to maximize generation performance of the trained model; this is also the same objective used in noise conditional score networks based on score matching. From a trained model, $\mathbf{x}_0$ is sampled by first sampling $\mathbf{x}_T$ from the prior $p_\theta(\mathbf{x}_T)$, and then sampling $\mathbf{x}_{t-1}$ from the generative processes iteratively. <br>
The length $T$ of the forward process is a important hyperparameter in DDPMs. From a variational perspective, a large T allows the reverse process to be close to a Gaussian, so that the generative process modeled with Gaussian conditional distributions becomes a good approximation; this motivates the choices of large $T$ values, such as $T=1000$. However, as all $T$ iterations have to be performed sequentially, instead of in parallel, to obtain a sample $\mathbf{x}_0$, sampling from DDPMs is much slower than sampling from other deep generative models, which makes them impractical for tasks where compute is limited and latency is critical. 




### 3 - Variational Inference For Non-Markovian Forward Processes

Because the generative model approximates the reverse of the inference process, we need to rethink the inference process in order to reduce the number of iterations required by the generative model. Our key observation is that the DDPM objective in the form of $L_\gamma$ only depends on the marginals $q(\mathbf{x}_t | \mathbf{x}_0)$, but not directly on the joint $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$. Since there are many inference distributions (joints) with the same marginals, we explore alternative inference processes that are non-Markovian, which leads to new generative processes. These non-Markovian inference process lead to the same surrogate objective function as DDPM, as we will show below. In Appendix A, we show that the non-Markovian perspective also applies beyond the Gaussian case. 

**Notes:**
- **The objective never sees the joint.** Each term just draws a noisy sample at timestep $t$ independently. You will never need $\mathbf{x}_1, \mathbf{x}_2, ...$ together - just $\mathbf{x}_t$ alone. The joint $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$, which describes how all timesteps relate to each other, never appears
- **There exist many joints with the same marginals.** The marginal $q(\mathbf{x}_t|\mathbf{x}_0)$ tells you the distribution of $\mathbf{x}_t$ given $\mathbf{x}_0$. But it says nothing about how $\mathbf{x}_t$ and $\mathbf{x}_{t-1}$ are correlated. You could have wildly different joints, different correlation structures between timesteps, and still get the same marginals. DDPM picked the simplest one: the Markovian joint where $\mathbf{x}_t$ only depends on $\mathbf{x}_{t-1}$. 
- **The trained model doesn't care about the joint**. Since $\epsilon_\theta$ is trained purely on $(\mathbf{x}_t, t)$ pairs sampled from the marginals and NOT the joints, it has no idea of which joint produced the samples. So you can swap the joint at inference time without touching the model. 
- **DDIM defines a whole new family of joints with the same marginal as DDPM**. DDIM introduces a family $\mathcal{Q}$ of joints indexed by $\sigma$, all satisfying: $$q_\sigma(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_0, (1-\alpha_t)\mathbf{I})\quad\forall t$$ Same marginals -> same $L_\gamma$ -> same trained $\epsilon_\theta$ works for every member of the family. DDPM ($\sigma_t>0$, Markovian) and DDIM ($\sigma_t=0$, deterministic) are just two points in this family. Fast sampling is simply choosing a member of $\mathcal{Q}$ whose reverse process requires fewer steps. 

#### 3.1 - Non-Markovian Forward Processes

Let us consider a family $\mathcal{Q}$ of inference distributions, indexed by a real vector $\sigma\in \R^T_{\geq 0}$: $$q_\sigma(\mathbf{x}_{1:T}|\mathbf{x}_0):=q_\sigma(\mathbf{x}_T|\mathbf{x}_0)\prod_{t=2}^T q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$$ where $q_\sigma(\mathbf{x}_T|\mathbf{x}_0)=\mathcal{N}(\sqrt{\alpha_T}\mathbf{x}_0, (1-\alpha_T)\mathbf{I})$ and for all $t>1$, $$q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)=\mathcal{N}\left(\sqrt{\alpha_{t-1}}\mathbf{x}_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\frac{\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_0}{\sqrt{1-\alpha_t}},\sigma_t^2\mathbf{I}\right).$$ The mean function is chosen to order to ensure that $q_\sigma(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_0, (1-\alpha_t)\mathbf{I})$ for all $t$, so it defines a joint inference distribution that matches the "marginals" as desired. The forward process can be derived from Bayes' rule: $$q_\sigma(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)=\frac{q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)q_\sigma(\mathbf{x}_t|\mathbf{x}_0)}{q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_0)},$$ which is Gaussian (although we do not use this fact for the remainder of this paper). Unlike the diffusion process in Eq. (3), the forward process here is no longer Markovian, since each $\mathbf{x}_t$ could depend on both $\mathbf{x}_{t-1}$ and $\mathbf{x}_0$. The magnitude of $\sigma$ controls how stochastic the forward process is; when $\sigma\rightarrow \mathbf{0}$, we reach an extreme case where as long as we observe $\mathbf{x}_0$ and $\mathbf{x}_t$ for some $t$, then $\mathbf{x}_{t-1}$ become known and fixed.  

**Notes:**
- **The joint is defined backwards intentionally**. Rather than building the joint forward (each step depends on the previous), DDIM defines it by specifying the reverse conditionals $q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ directly, anchored at $q_\sigma(\mathbf{x}_T|\mathbf{x}_0)$. This is valid by the chain rule - any joint can be factored this way. 
- **The reverse conditional is engineered to preserve marginals**. The mean has two components. The first term $\sqrt{\alpha_{t-1}}\mathbf{x}_0$ points towards the clean image. The second term $\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\frac{\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_0}{\sqrt{1-\alpha_t}}$ is the direction pointing towards $\mathbf{x}_t$. Specifically, it is the noise component of $\mathbf{x}_t$. The coefficients are carefully chosen so that when you maginalize out $\mathbf{x}_{t-1}$. you recover exactly $q_\sigma(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_0, \sigma_t^2\mathbf{I})$, the DDPM marginal. 
- **$\sigma$ controls stochasticity.** The variance $\sigma_t^2\mathbf{I}$ is a free parameter. When $\sigma_t>0$, sampling $\mathbf{x}_{t-1}$ from this conditional is stochastic - you add random noise. As $\sigma\rightarrow 0$, the variance vanishes and the mean fully determines $\mathbf{x}_{t-1}$ given $\mathbf{x}_t$ and $\mathbf{x}_0$ and the process becomes fully deterministic. This is the DDIM sampler. 
- **The forward process is non-Markovian**. Notice the conditioning in the recovered forwards direction. The forward step $\mathbf{x}_t$ depends on both $\mathbf{x}_{t-1}$ and $\mathbf{x}_0$, not just $\mathbf{x}_{t-1}$. This is what makes it non-Markovian. But this doesn't matter for training as the forward process is never directly used. Only the marginals are used, and those are identical to DDPM by construction. 

<br>

#### 3.2 - Generative Process and Unified Variational Inference Objective

Next, we define a trainable generative process $p_\theta(\mathbf{x}_{0:T})$ where each $p_\theta^{(t)}(\mathbf{x}_{t-1}|\mathbf{x}_t)$ leverages knowledge of $q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$. Intuitively, given a noisy observation $\mathbf{x}_t$, we first make a prediction of the corresponding $\mathbf{x}_0$, and then use it to obtain a sample $\mathbf{x}_{t-1}$ through the reverse conditional distribution $q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$, which we have defined. 

For some $\mathbf{x}_0\sim q(\mathbf{x}_0)$ and $\epsilon_t\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$, $\mathbf{x}_t$ can be obtained using Eq. (4). The model $\epsilon_\theta^{(t)}(\mathbf{x}_t)$ then attempts to predict $\epsilon_t$ from $\mathbf{x}_t$, without knowledge of $\mathbf{x}_0$. By rewriting Eq. (4), one can then predict the denoised observation, which is a prediction of $\mathbf{x}_0$ given $\mathbf{x}_t$: $$f_\theta^{(t)}(\mathbf{x}_t):=(\mathbf{x}_t-\sqrt{1-\alpha_t}\cdot\epsilon_\theta^{(t)}(\mathbf{x}_t))/\sqrt{\alpha_t}$$ We can then define the generative process with a fixed prior $p_\theta(\mathbf{x}_T)=\mathcal{N}(\mathbf{0}, \mathbf{I})$ and $$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=\begin{cases} \mathcal{N}(f_\theta^{(1)}(\mathbf{x}_1), \sigma_1^2 \mathbf{I}) & \text{if }t=1 \\ q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, f_\theta^{(t)}(\mathbf{x}_t)) & \text{otherwise}\end{cases}$$ where $q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, f_\theta^{(t)}(\mathbf{x}_t))$ is defined as in Eq. (7) with $\mathbf{x}_0$ replaced by $f_\theta^{(t)}(\mathbf{x}_t)$. We add some Gaussian noise (with covariance $\sigma_1^2\mathbf{I}$) for the case of $t=1$ to ensure that the generative process is supported everywhere. 

**Notes:**
- **Predict $\mathbf{x}_0$ first, then step backwards**. DDPM's reverse process directly models $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ as a Gaussian with a learned mean. DDIM instead breaks this into two explicit step: 
1. Use $\epsilon_\theta^{(t)}(\mathbf{x}_t)$ to predict the noise, then back out a prediction of $\mathbf{x}_0$: $$f_\theta^{(t)}(\mathbf{x}_t):=\frac{\mathbf{x}_t-\sqrt{1-\alpha_t}\cdot\epsilon_\theta^{(t)}(\mathbf{x}_t)}{\sqrt{\alpha_t}}$$ This is just rearranging $\mathbf{x}_t=\sqrt{\alpha_t}\mathbf{x}_0+\sqrt{1-\alpha_t}\epsilon_t$ to solve for $\mathbf{x}_0$. So $f_\theta^{(t)}(\mathbf{x}_t)$ is the model's best guess of the clean image given the current noisy sample. 
2. Plug that predicted $\mathbf{x}_0$ into the reverse conditional $q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$, substituting $f_\theta^{(t)}(\mathbf{x}_t)$ int place of the true $\mathbf{x}_0$: $$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t,f_\theta^{(t)}(\mathbf{x}_t))$$
- This is powerful because the reverse conditional $q_\sigma$ was designed so that its mean interpolates between the predicted $\mathbf{x}_0$ and the current $\mathbf{x}_t$. When $\sigma_t=0$, this becomes fully deterministic. Given $\mathbf{x}_t$ and a predicted $\mathbf{x}_0$, the next sample $\mathbf{x}_{t-1}$ is completely fixed with no randomness added. This is what makes DDIM a deterministic sampler.
- **The $t=1$ special case**: At the final step, rather than using $q_\sigma$, the model just outputs a Gaussian centered at $f_\theta^{(1)}(\mathbf{x}_1)$ with small variance $\sigma_1^2\mathbf{I}$. This is a technicality to ensure the generative distribution has full support everywhere - without it, the distribution could be degenerate (a delta function when $\sigma\rightarrow 0$), which causes issues with likelihood computation. 
- **The big picture**: The generative process is now: start from $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$, then at each step predict the clean image, use that prediction to compute where $\mathbf{x}_{t-1}$ should be, and repeat. The model $\epsilon_\theta$ is the same one trained with the DDPM objective. Nothing changes about training, only about how you use the model at inference time. 

<br>

We optimize $\theta$ via the following variational inference objective (which is a functional over $\epsilon_\theta$): $$J_\sigma(\epsilon_\theta):=\mathbb{E}_{\mathbf{x}_{0:T}\sim q_\sigma(\mathbf{x}_{0:T})}[\log q_\sigma(\mathbf{x}_{1:T}|\mathbf{x}_0)-\log p_\theta(\mathbf{x}_{0:T})]$$ $$=\mathbb{E}_{\mathbf{x}_{0:T}\sim q_\sigma(\mathbf{x}_{0:T})}\left[\log q_\sigma(\mathbf{x}_T|\mathbf{x}_0)+\sum_{t=2}^T\log q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)-\sum_{t-1}^T \log p_\theta^{(t)}(\mathbf{x}_{t-1}|\mathbf{x}_t)-\log p_\theta(\mathbf{x}_T)\right]$$ where we factorize $q_\sigma(\mathbf{x}_{1:T}|\mathbf{x}_0)$ according to Eq. (6) and $p_\theta(\mathbf{x}_{0:T})$ according to Eq. (1). 

From the definition of $J_\sigma$, it would appear that a different model has to be trained for every choice of $\sigma$ since it corresponds to a different variational objective (and a different generative process). However, $J_\sigma$ is equivalent to $L_\gamma$ for certain weights $\gamma$, as we show below.

**Theorem 1.** *For all $\sigma>\mathbf{0}$, there exists $\gamma\in\R^T_{>0}$ and $C\in\R$, such that $J_\sigma=L_\gamma+C$*.

The variational objective $L_\gamma$ is special in the sense that if parameters $\theta$ of the models $\epsilon_\theta^{(t)}$ are not shared across different $t$, then the optimal solution for $\epsilon_\theta$ will not depend on the weights $\gamma$ (as global optimum is achieved by separately maximizing each term in the sum). This property of $L_\gamma$ has two implications. On one hand, this justified the use of $L_1$ as a surrogate objective function for the variation lower bound in DDPMs; on the other hand, since $J_\sigma$ is equivalent to some $L_\gamma$ from Theorem 1, the optimal solution of $J_\sigma$ is the same as that of $L_1$. Therefore, if parameters are not shared across $t$ in the model $\epsilon_\theta$, then the $L_1$ objective used by Ho et al. (2020) can be used as a surrogate objetive for the variational objective $J_\sigma$ as well. 

**Notes:**
- The objective $J_\sigma$ is just the standard ELBO in disguise. It's just the KL divergence between the inference process $q_\sigma$ and the generative process $p_\theta$, which is the standard variational inference objective - minimizing this forces $p_\theta$ to match $q_\sigma$. The second line just expands both using their factorizations, giving you a sum of log terms - each term is a KL divergence between $q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ and $p_\theta^{(t)}(\mathbf{x}_{t-1}|\mathbf{x}_t)$ at each timestep. 
- **This seems like a problem.** Since $J_\sigma$ depends on $\sigma$, it looks like you'd need to retrain $\epsilon_\theta$ for every choice of $\sigma$ - every different sampler would need its own model. That would defeat the entire purpose. 
- **Theorem 1 kills this problem.** It says that for any $\sigma>0$, $J_\sigma$ differs from $L_\gamma$ only by a constant $C$ and a reweighting $\gamma$ of the timestep terms: $$J_\sigma=L_\gamma+C$$ The constant $C$ doesn't affect optimization. So minimizing $J_\sigma$ over $\theta$ is equivalent to minimizing $L_\gamma$ over $\theta$ - same optimal parameters regardless of $\sigma$.
- **The weights $\gamma$ don't matter**. This is the subtle but crucial point. If the parameters $\theta^{(t)}$ are not shared across timesteps, meaning each $\epsilon_\theta^{(t)}$ is an independent function, then $L_\gamma$ decomposes into $T$ independent terms: $$L_\gamma=\sum_{t=1}^T\gamma_t\cdot\underbrace{\mathbb{E}\left[\|\epsilon_\theta^{(t)}(\mathbf{x}_t)-\epsilon_t\|^2\right]}_\text{optimized independently}$$ Each terms can be minimized independently, and $\gamma_t>0$ just scales the gradient but doesn't change where the minimum is. So the optimal $\epsilon_\theta^{(t)}$ is the same regardless of what $\gamma_t$ is. This means $L_1$ (setting all $\gamma_t=1$, as Ho et al. do) and $L_\gamma$ for any positive $\gamma$ all share the same optimum. 
- Chaining these facts together: $J_\sigma\equiv L_\gamma$ (Theorem 1), and $L_\gamma\equiv L_1$ at the optimum (weight invariance). Therefore the model trained with the simple $L_1$ DDPM objective is simultaneously the optimal model for $J_\sigma$ for every valid $\sigma$. You train once, then freely choose $\sigma$ and hence the sampler at inference time. 
- In practice, modern diffusion models do share parameters across $t$ (via a single network conditioned on $t$), which technically breaks the weight-invariance argument. But empirically it still holds well enough, which is why the same pretrained DDPM checkpoint works across different DDIM sampling schedules without any fine-tuning. 

<br>

### 4 - Sampling From Generalized Generative Process

With $L_1$ as the objective, we are not only learning a generative process for Markovian inference process, but also generative processes for many non-Markovian forward processes parameterized by $\sigma$ that we have described. Therefore, we can essentially use pretrained DDPM models as the solutions to the new objectives, and focus on finding a generative process that is better at producing samples subject to our needs by changing $\sigma$.

#### 4.1 - Denoising Diffusion Implicit Models

From $p_\theta(\mathbf{x}_{1:T})$ in Eq. (10), one can generate a sample $\mathbf{x}_{t-1}$ from a sample $\mathbf{x}_t$ via: $$\mathbf{x}_{t-1}=\sqrt{\alpha_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t-\sqrt{1-\alpha_t}\epsilon_\theta^{(t)}(\mathbf{x}_t)}{\sqrt{\alpha_t}}\right)}_{\text{"predicted }\mathbf{x}_0\text{"}}+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta^{(t)}(\mathbf{x}_t)}_{\text{"direction pointing to }\mathbf{x}_t\text{"}}+\underbrace{\sigma_t\epsilon_t}_\text{random noise}$$ where $\epsilon_t\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$ is standard Gaussian noise independent of $\mathbf{x}_t$, and we define $\alpha_0:= 1$. Different choices of $\sigma$ values results in different generative processes, all while using the same model $\epsilon_\theta$, so re-training the model is unnecessary. When $\sigma_t=\sqrt{(1-\alpha_{t-1})/(1-\alpha_t)}\sqrt{1-\alpha_t/\alpha_{t-1}}$ for all $t$, the forward process becomes Markovian, and the generative process becomes a DDPM. 

We note another special case when $\sigma_t=0$ for all $t$; the forward process becomes deterministic given $\mathbf{x}_{t-1}$ and $\mathbf{x}_0$, except for $t=1$; in the generative process, the coefficient before the random noise $\epsilon_t$ becomes zero. The resulting model becomes an implicit probabilistic model, where samples are generated from latent variables with a fixed procedure (from $\mathbf{x}_T$ to $\mathbf{x}_0$). We name this the *denoising diffusion implicit model* (DDIM), because it is an implicit probabilistic model trained with the DDPM objective (despite the forward process no longer being a diffusion). 

