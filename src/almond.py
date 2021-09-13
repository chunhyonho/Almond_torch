import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .utils import EMISSION


def random_vector(dim):
    x = torch.randn(dim)
    return x


def to_leaf(tensor):
    t = tensor.detach().clone()
    t.requires_grad = True
    return t


class ALMOND(pl.LightningModule):

    def __init__(
            self,
            encoder_mu: nn.Module,
            encoder_logvar: nn.Module,
            decoder: nn.Module,
            step_size: float,
            total_step: int,
            batch_size: int,
            num_data: int
    ):
        super().__init__()
        self.save_hyperparameters()

        for param in encoder_mu.parameters():
            param.requires_grad = False

        for param in encoder_logvar.parameters():
            param.requires_grad = False

        self.decoder = decoder

        self.emission = EMISSION(
            distribution='Poisson'
        )

        self.particle_dict = {
            f'particle_{i}': random_vector(self.decoder.input_dim) for i in range(num_data)
        }

        self.automatic_optimization = False

    def forward(self, z):
        return self.decoder(z)

    def get_init_particle(self, x):
        mu = self.encoder_mu(x)
        std = torch.exp(self.encoder_logvar(x) / 2)

        # sample particle for z_init
        particle = torch.distributions.Normal(mu, std).rsample()
        particle.requires_grad = True

        return particle

    def prior(self, z):
        return - z ** 2 / 2

    def log_likelihood(self, x, z):
        recon_x = self(z)

        log_pz = self.prior(z).sum(-1).mean()
        log_pxz = self.emission(recon_x).log_prob(x).sum(-1).mean()

        return log_pz + log_pxz

    def langevin_sample(self, x, particle):
        score = self.grad_with_z(x, particle)
        return particle + self.hparams.step_size * score + math.sqrt(2 * self.hparams.step_size) * torch.randn(
            particle.shape, device=self.device)

    def grad_with_params(self, x, particle):
        log_likelihood = self.log_likelihood(x, particle)
        log_likelihood.backward()
        return [- param.grad.detach() for param in self.parameters()]

    def grad_with_z(self, x, particle):
        log_likelihood = self.log_likelihood(x, particle)
        log_likelihood.backward(inputs=[particle])
        return particle.grad.detach()

    def training_step(self, batch, batch_idx):
        x, y, idx = batch

        particle = torch.stack([self.particle_dict[f'particle_{i}'] for i in idx]).to(self.device)
        particle.requires_grad = True

        opt = self.optimizers()
        opt.zero_grad()

        mc_grad = self.grad_with_params(x, particle)
        particle = to_leaf(self.langevin_sample(x, particle))

        for _ in range(self.hparams.total_step):
            mc_grad = [sum_grad + grad for sum_grad, grad in zip(mc_grad, self.grad_with_params(x, particle))]
            particle = to_leaf(self.langevin_sample(x, particle))

        with torch.no_grad():
            for param, grad in zip(self.parameters(), mc_grad):
                param.grad = grad / self.hparams.total_step

        opt.step()

        for i, idx in enumerate(idx):
            self.particle_dict[f'particle_{i}'] = particle[i].detach().cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=3e-4)
