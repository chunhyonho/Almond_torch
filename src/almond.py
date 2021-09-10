
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import pytorch_lightning as pl

from .utils import EMISSION


def PRIOR(distribution: str, latent_dim: int):
    if distribution == 'Gaussian':
        return torch.distributions.Normal(
            torch.zeros(latent_dim).cuda(),
            torch.ones(latent_dim).cuda()
        )


def random_vector(dim):
    x = torch.randn(dim)
    x.requires_grad = True
    return x


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

        self.prior = PRIOR(
            distribution='Gaussian',
            latent_dim=self.decoder.input_dim
        )

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

    def log_likelihood(self, x, z):
        recon_x = self(z)

        log_pz = self.prior.log_prob(z).sum(-1).mean()
        log_pxz = self.emission(recon_x).log_prob(x).sum(-1).mean()

        return log_pz + log_pxz

    def langevin_sample(self, x, particle):
        score = self.grad_with_z(x, particle)
        return particle + self.hparams.step_size * score + math.sqrt(2 * self.hparams.step_size) * torch.randn(
            particle.shape)

    def grad_with_params(self, x, particle):
        log_likelihood = self.log_likelihood(x, particle)
        log_likelihood.backward()
        return [- param.grad for param in self.parameters()]

    def grad_with_z(self, x, particle):
        log_likelihood = self.log_likelihood(x, particle)

        log_likelihood.backward(retain_graph=True, inputs=[particle])
        return particle.grad

    def training_step(self, batch, batch_idx):
        x, y, idx = batch

        particle = torch.stack([self.particle_dict[f'particle_{i}'] for i in idx]).cuda()

        opt = self.optimizers()
        opt.zero_grad()

        mc_grad = self.grad_with_params(x, particle)
        particle = self.langevin_sample(x, particle)

        for _ in range(self.hparams.total_step):
            mc_grad = [sum_grad + grad for sum_grad, grad in zip(mc_grad, self.grad_with_params(x, particle))]
            particle = self.langevin_sample(x, particle)

        with torch.no_grad():
            for param, grad in zip(self.parameters(), mc_grad):
                param.grad = grad / self.hparams.total_step

        opt.step()

        for i in idx:
            self.particle_dict[f'particle_{i}'] = particle[i]

        x_hat = self.decoder(particle)

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=3e-4)
