import torch
import math

class DiffusionScheduler:
    def __init__(self, device, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.T, (batch_size,), device=device)

    def add_noise(self, x_start, t, noise):
        """
        x_start: [B, T_future, D]
        noise: same shape
        """
        alpha_bar = self.alpha_cumprod[t].view(-1, 1, 1).to(x_start.device)

        return (
            torch.sqrt(alpha_bar) * x_start +
            torch.sqrt(1 - alpha_bar) * noise
        )

