import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .distributions import EMISSION


def random_vector(shape):
    x = torch.randn(shape)
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


def tile(tensor, n):
    """
    tile tensor n times,
    tensor : d1 x d2 x ... x dN -> n x d1 x ... x dN
    """
    return tensor.repeat((n,) + (1,) * tensor.dim())


class ALMOND(pl.LightningModule):

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            step_size_init: float,
            total_step: int,
            num_chain: int,
            batch_size: int,
            num_train_data: int,
            min_lr: float,
            max_lr: float,
            validation_total_step: int
    ):
        super().__init__()
        self.save_hyperparameters(
            "step_size_init",
            "total_step",
            "num_chain",
            "batch_size",
            "num_train_data",
            "min_lr",
            "max_lr",
            "validation_total_step"
        )

        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.decoder = decoder

        self.emission = EMISSION(
            distribution='Normal'
        )

        self.particle_dict = dict()

        self.automatic_optimization = False

    def forward(self, z):
        return self.decoder(z)

    def std_gaussian_likelihood(self, obs, mu):
        return torch.distributions.Normal(mu, torch.ones_like(mu)).log_prob(obs)

    def log_likelihood(self, x, z):
        """
        x : { ( num_chain ) * batch_size } x d1 x ... x dN
        z : { ( num_chain ) * batch_size } x d
        """

        recon_x = self(z)

        log_pz = self.std_gaussian_likelihood(obs=z, mu=torch.zeros_like(z)).sum(dim=-1)

        if z.dim() == 2:
            log_pxz = self.std_gaussian_likelihood(obs=x, mu=recon_x).sum(dim=tuple(range(1, x.dim())))
            return (log_pz + log_pxz).sum()
        else:
            log_pxz = self.std_gaussian_likelihood(obs=x, mu=recon_x).sum(dim=tuple(range(2, x.dim())))
            return (log_pz + log_pxz).sum()

    def langevin_chain(self, x, particle):
        score = self.grad_with_z(x, particle)
        return particle + self.current_step_size * score + math.sqrt(2.0 * self.current_step_size) * torch.randn_like(
            particle)

    def grad_with_params(self, x, particle):
        """
        It returns '- d (log_likelihood) / d (param)'
        """
        log_likelihood = self.log_likelihood(x, particle) /(self.hparams.num_chain * self.hparams.batch_size)
        log_likelihood.backward()
        return [negative_grad_detacher(name_param) for name_param in self.named_parameters()]

    def grad_with_z(self, x, particle):
        """
        It returns 'd (log_likelihood) / dW'
        """
        log_likelihood = self.log_likelihood(x, particle)
        log_likelihood.backward(inputs=[particle])
        return particle.grad.detach()

    @property
    def current_step_size(self):
        return max(self.hparams.step_size_init / 10.0,
                   self.hparams.step_size_init / math.pow(self.current_epoch + 1.0, 0.5))

    def training_step(self, batch, batch_idx):
        x, y, indices = batch

        last_particles = []
        for i, idx in enumerate(indices):
            try:
                last_particle = self.particle_dict[f'particle_{idx}'].to(self.device)
            except KeyError:
                mu, log_var = self.encoder(x[i])

                last_particle = torch.distributions.Normal(
                    mu, torch.exp(log_var / 2)
                ).rsample((self.hparams.num_chain,))

                self.particle_dict[f'particle_{idx}'] = last_particle.cpu()

            last_particles.append(last_particle)

        x_tiled = tile(x, self.hparams.num_chain)

        # particle : num_chain x batch_size x latent_dim
        particle = torch.stack(last_particles, dim=1).to(self.device)
        particle.requires_grad = True

        opt = self.optimizers()
        opt.zero_grad()

        mc_grad = self.grad_with_params(x_tiled, particle)
        particle = to_leaf(self.langevin_chain(x_tiled, particle))

        for _ in range(self.hparams.total_step - 1):
            mc_grad = [grad_sum(sum_grad, grad) for sum_grad, grad in zip(mc_grad, self.grad_with_params(x_tiled, particle))]
            particle = to_leaf(self.langevin_chain(x_tiled, particle))

        with torch.no_grad():
            for param, grad in zip(self.parameters(), mc_grad):
                if grad is not None:
                    param.grad = grad / self.hparams.total_step
                else:
                    param.grad = None

        opt.step()

        # save the multi-chain langevin samples
        for i, idx in enumerate(indices):
            self.particle_dict[f'particle_{idx}'] = particle[:, i].detach().cpu()

        x_hat = self(particle).mean(dim=0)  # n_chain x batch_size x data_dim
        # recon_loss = - (self.emission(x_hat).log_prob(x)).sum(dim=-1).mean()

        recon_loss = F.mse_loss(
            x_hat, x, reduction='none'
        ).sum(dim=tuple(range(1, x.dim()))).mean() / 2

        self.log(f"train_recon_loss", recon_loss, on_step=True, on_epoch=False, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        x, y, _ = batch

        particle, _ = self.encoder(x)
        particle.requires_grad = True

        particles = []

        for i in range(self.hparams.validation_total_step):
            particle = to_leaf(self.langevin_chain(x, particle))

            if i >= self.hparams.validation_total_step - 100:
                particles.append(particle)

        particles = torch.stack(particles)

        x_hat = self(particles).mean(dim=0)  # batch_size x data_dim
        # recon_loss = - (self.emission(x_hat).log_prob(x)).sum(dim=-1).mean()
        recon_loss = F.mse_loss(
            x_hat, x, reduction='none'
        ).sum(dim=tuple(range(1, x.dim()))).mean() / 2

        self.log(f"val_recon_loss", recon_loss, on_epoch=True, sync_dist=True, prog_bar=True)

        torch.set_grad_enabled(False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.hparams.max_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, eta_min=self.hparams.min_lr
                )
            }
        }