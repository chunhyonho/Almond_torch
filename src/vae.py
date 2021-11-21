from math import sqrt
import io
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from PIL import Image
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
            num_latent_variable_samples: int,
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
        return self.decoder(z).mean(dim=0), mu, std

    def sample(self, mu, std):
        return torch.distributions.Normal(mu, std).rsample((self.hparams.num_latent_variable_samples, ))

    def step(self, x):
        x_hat, mu, std = self._run_step(x)
        recon_loss = F.mse_loss(
            x_hat.flatten(start_dim=1), x.flatten(start_dim=1), reduction='none'
        ).sum(dim=1).mean()

        kl = 0.5 * torch.sum(mu ** 2 + std ** 2 - 2 * torch.log(std + 1e-10) - 1, 1).mean()

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "loss": loss,
        }
        return loss, logs, x_hat, mu

    def training_step(self, batch, batch_idx):
        X, y, _ = batch
        loss, logs, _, _ = self.step(X)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, _ = batch
        loss, logs, x_hat, mu = self.step(X)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_epoch=True, sync_dist=True, prog_bar=True)

        if self.current_epoch % 10 == 0:
            if batch_idx == 0 and self.hparams.image:

                image_dim = x_hat.size(-1)

                image_pair = torch.cat([X[:6], x_hat[:6]], dim=-1).reshape(
                    -1, 1, int(sqrt(image_dim)) * 2, int(sqrt(image_dim))
                )

                self.logger.log_image(
                    key="Reconstruction",
                    images=[make_grid(image_pair, nrow=2, normalize=True)]
                )

        return mu, y

    def validation_epoch_end(self, outputs) -> None:

        if self.current_epoch % 10 == 0:

            latents = []
            labels = []

            for z, y in outputs:
                latents.append(z)
                labels.append(y)

            latents = torch.stack(latents).cpu()
            labels = torch.stack(labels).cpu()

            plt.figure(figsize=(8, 8))
            for c in torch.unique(labels):
                z_c = latents[labels == c]
                plt.scatter(z_c[:, 0], z_c[:, 1], label=f'{int(c)}')

            plt.legend()

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            im = Image.open(img_buf)

            self.logger.log_image(
                key="Scatter Plot", images=[im]
            )
            img_buf.close()
            plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
