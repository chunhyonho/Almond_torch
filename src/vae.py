import pytorch_lightning as pl

from .layers import *
from torch.nn import functional as F

from typing import Sequence


def ENCODER(model: str, input_dim: int, output_dim: int, hidden_dim: Sequence[int] = [], **kwargs):
    if model == 'MLP':
        return MLP(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
        )


def DECODER(model: str, input_dim: int, output_dim: int, hidden_dim: Sequence[int] = [], **kwargs):
    if model == 'MLP':
        return MLP(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
        )


class VAE(pl.LightningModule):

    def __init__(
            self,
            latent_dim: int,
            output_dim: int,
            encoder: str,
            decoder: str,
            learning_rate:float,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_mu = ENCODER(
            model=encoder,
            input_dim=output_dim,
            output_dim=latent_dim,
            hidden_dim=kwargs.get('hidden_dim')
        )

        self.encoder_logvar = ENCODER(
            model=encoder,
            input_dim=output_dim,
            output_dim=latent_dim,
            hidden_dim=kwargs.get('hidden_dim')
        )

        self.decoder = DECODER(
            model=decoder,
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=kwargs.get('hidden_dim')[::-1]
        )


    def from_pretrained(self, checkpoint_name):
        return self.load_from_checkpoint(checkpoint_name, strict=False)

    def forward(self, x):
        mu = self.encoder_mu(x)
        log_var = self.encoder_logvar(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        mu = self.encoder_mu(x)
        log_var = self.encoder_logvar(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
