# ml-lab

A place where I reimplement machine learning papers from scratch to understand
them properly. Reading a paper and being able to build the thing turn out to be
different skills, and this repo is me closing that gap one paper at a time. The
rule I set for myself is to write the core of each model or optimizer by hand
first, not just call a library, then run it on real data and report honest
numbers (including the parts that didn't work).

## Projects

| Paper | Code | What I did |
|---|---|---|
| [Q-learning](https://link.springer.com/article/10.1007/BF00992698) (Watkins & Dayan, 1992) | [q-learning/](projects/q-learning) | Tabular Q-learning vs SARSA and Expected SARSA on cliff walking, then the same update rule playing snake, pong and flappy bird; pong and flappy get solved, snake plateaus at 21 because its state can't see its own body |
| [DQN](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013) | [deep-q-learning/](projects/deep-q-learning) | DQN with experience replay and a target network on my grid snake and breakout; breaks the tabular snake plateau (27.7 vs 21.1 foods) and clears breakout perfectly, at the cost of 1M environment steps |
| [Adam](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014) | [adam/](projects/adam) | Adam written from scratch, benchmarked against SGD with momentum on CIFAR-10 with a ResNet-18 |
| [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215) (Sutskever et al., 2014) | [machine-translation/](projects/machine-translation) | LSTM encoder-decoder trained on all 40.8M WMT14 en-fr pairs, 14.5 BLEU with beam search on newstest2013 |
| [PPO](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) | [ppo/](projects/ppo) | PPO from scratch on MuJoCo Humanoid-v5; learns a six-second lurching run (return 2147, best 4103) in 20M steps, with the mid-training KL blowup and noisy plateau reported rather than smoothed away |
| [DDPM](https://arxiv.org/abs/2006.11239) (Ho et al., 2020) | [ddpm/](projects/ddpm) | U-Net noise predictor with a linear schedule, trained on CelebA 64x64 |
| [DDIM](https://arxiv.org/abs/2010.02502) (Song et al., 2020) | [ddim/](projects/ddim) | Faster non-Markovian sampling on top of DDPM (notes so far, still in progress) |
| [ALiBi](https://arxiv.org/abs/2108.12409) (Press et al., 2021) | [positional-encodings/](projects/positional-encodings) | Compared learned / RoPE / ALiBi / NoPE on length extrapolation; ALiBi stays flat out to 8k while the others blow up past their 512 training length |
| [Switch Transformer](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021) | [switch-transformer/](projects/switch-transformer) | Top-1 mixture of experts at matched FLOPs; switch-8 beats the dense baseline by 1.9 ppl and the experts specialize into dates, names, units and modal verbs on their own |
| [Muon](https://kellerjordan.github.io/posts/muon/) (Jordan, 2024) | [muon/](projects/muon) | Muon optimizer with Newton-Schulz orthogonalization, compared against AdamW on a GPT-2 small (85M) trained on FineWeb-Edu |
| [SDFT](https://arxiv.org/abs/2601.19897) (Shenfeld et al., 2026) | [self-distillation/](projects/self-distillation) | On-policy self-distillation for injecting post-cutoff facts into Qwen2.5-3B; the knowledge result did not reproduce at 3B (SFT learned ~2x more), the retention claim did directionally, and the audit that established both is the interesting part |
| [Bradley-Terry reward model](https://arxiv.org/abs/2203.02155) (Bradley & Terry, 1952; Ouyang et al., 2022) | [reward-modelling/](projects/reward-modelling) | Scalar-head reward models trained on 59k UltraFeedback preference pairs; Qwen2.5-0.5B hits 81.2% held-out pairwise accuracy and Qwen3.5-0.8B hits 84.6%, and the length bias is measured rather than ignored (it shrinks with the better backbone but does not go away) |

## Setup

Each project is self-contained under `projects/<name>` with its own README and,
once there's code to run, a `requirements.txt`. Pick one and install what it
needs:

```bash
cd projects/machine-translation
pip install -r requirements.txt
```

If you're on a Blackwell GPU (RTX 5090 and similar), stable PyTorch may not
support it yet, so install the nightly build:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## License

MIT, see [LICENSE](LICENSE).
