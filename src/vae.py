import pytorch_lightning as pl

from torch.nn import functional as F

from .layers import *
from .distributions import EMISSION

from typing import Sequence


def ENCODER(model: str, input_dim: int, output_dim: int, hidden_dim: Sequence[int] = [], **kwargs):
    class ENC(nn.Module):
        def __init__(self, shared_layer, latent_dim):
            super().__init__()

            self.layer = shared_layer
            self.fc_mu = nn.Linear(self.layer.output_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.layer.output_dim, latent_dim)

        def forward(self, x):
            x = self.layer(x)

            mu = self.fc_mu(x)
            log_var = self.fc_logvar(x)

            return mu, log_var

    if model == 'MLP':
        return ENC(
            shared_layer=MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim
            ),
            latent_dim=output_dim
        )


def DECODER(
        model: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: Sequence[int] = [],
        positive_constraint: bool = False,
        **kwargs
):
    hidden_dim = list(chain(hidden_dim, [output_dim]))
    if model == 'MLP':
        return MLP(
            input_dim=input_dim, hidden_dim=hidden_dim, positive=positive_constraint
        )


class VAE(pl.LightningModule):

    def __init__(
            self,
            latent_dim: int,
            output_dim: int,
            encoder_architecture: str,
            encoder_hidden_dim: list,
            decoder_architecture: str,
            decoder_hidden_dim: list,
            decoder_positive_constraint: bool,
            learning_rate: float,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder initialization
        self.encoder = ENCODER(
            model=encoder_architecture,
            input_dim=output_dim,
            output_dim=latent_dim,
            hidden_dim=encoder_hidden_dim
        )

        self.decoder = DECODER(
            model=decoder_architecture,
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=decoder_hidden_dim,
            positive_constraint=decoder_positive_constraint
        )

        self.emission = EMISSION(
            distribution='Normal'
        )

    def forward(self, x):
        mu, log_var = self.encoder(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        mu, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        z = self.sample(mu, std)
        return self.decoder(z), mu, std

    def sample(self, mu, std):
        z = mu + std * torch.randn_like(mu)
        return z

    def step(self, batch, batch_idx):
        x, y, _ = batch
        x_hat, mu, std = self._run_step(x)
        recon_loss = F.mse_loss(x_hat.flatten(start_dim=1), x.flatten(start_dim=1), reduction='none').sum(dim=1).mean() / 2

        kl = 0.5 * torch.sum(mu ** 2 + std ** 2 - 2 * torch.log(std + 1e-10) - 1, 1).mean()

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
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
