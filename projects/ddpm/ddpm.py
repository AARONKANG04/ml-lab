import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model
        self.T = T
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise=None):
        """Forward process: sample x_t from x_0 in closed form."""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def loss(self, x0):
        """Simplified training loss: MSE between true and predicted noise."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, shape):
        """Reverse process: generate samples starting from pure noise."""
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, t_batch)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x