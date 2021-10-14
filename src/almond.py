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


def negative_grad_detacher(x):
    name, param = x
    if not name.startswith('encoder.'):
        return - param.grad.detach()
    else:
        return None


def grad_sum(g1, g2):
    if g1 is not None and g2 is not None:
        return g1 + g2
    else:
        return None


class ALMOND(pl.LightningModule):

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            step_size: float,
            total_step: int,
            batch_size: int,
            num_data: int
    ):
        super().__init__()
        self.save_hyperparameters(
            "step_size",
            "total_step",
            "batch_size",
            "num_data"
        )

        self.encoder = encoder
        self.encoder.requires_grad_(False)
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

    def warm_up_setup(self, batch) -> None:
        x, y, idx = batch
        self.particle_dict[f'particle_{idx}'] = self.encoder(x)

    def get_init_particle(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(log_var / 2)

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
        return [negative_grad_detacher(name_param) for name_param in self.named_parameters()]

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

        for _ in range(self.hparams.total_step - 1):
            mc_grad = [grad_sum(sum_grad, grad) for sum_grad, grad in zip(mc_grad, self.grad_with_params(x, particle))]
            particle = to_leaf(self.langevin_sample(x, particle))

        with torch.no_grad():
            for param, grad in zip(self.parameters(), mc_grad):
                if grad is not None:
                    param.grad = grad / self.hparams.total_step
                else:
                    param.grad = None

        opt.step()

        for i, idx in enumerate(idx):
            self.particle_dict[f'particle_{i}'] = particle[i].detach().cpu()

        x_hat = self(particle)
        recon_loss = - (self.emission(x_hat).log_prob(x)).sum(dim=-1).mean()
        self.log(f"train_recon_loss", recon_loss, on_step=True, prog_bar=True, logger=True)

        return recon_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        x, y, _ = batch

        particle, _ = self.encoder(x)
        particle.requires_grad = True

        for _ in range(self.hparams.total_step):
            particle = to_leaf(self.langevin_sample(x, particle))

        x_hat = self(particle)
        recon_loss = - (self.emission(x_hat).log_prob(x)).sum(dim=-1).mean()
        self.log(f"val_recon_loss", recon_loss, prog_bar=True, logger=True, sync_dist=True)

        torch.set_grad_enabled(False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=3e-4)
