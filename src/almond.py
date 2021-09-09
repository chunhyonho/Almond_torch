import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

def PRIOR(distribution: str, latent_dim: int):
    if distribution == 'Gaussian':
        return torch.distributions.Normal(
            torch.zeros(latent_dim),
            torch.ones((latent_dim, latent_dim))
        )

def EMISSION(distribution: str):
    if distribution == 'Binomial':
        def rv(mean_prob: torch.Tensor):
            return torch.distributions.Binomial(
                mean_prob
            )

        return rv

    elif distribution == 'Gaussian':
        def rv(mean: torch.Tensor):
            return torch.distributions.Normal(
                mean, torch.diag_embed(torch.ones_like(mean))
            )

        return rv

class ALMOND(pl.LightningModule):

    def __init__(
            self,
            encoder_mu: nn.Module,
            encoder_logvar: nn.Module,
            decoder: nn.Module,
            step_size: float,
            total_step: int,
            batch_size: int
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
            distribution='Gaussian'
        )

        self.particle = torch.randn(
            (self.hparams.batch_size, self.decoder.input_dim),
            requires_grad=True
        )

        self.automatic_optimization = False

    def forward(self, z):
        return self.decoder(z)

    def log_likelihood(self, x, z):
        recon_x = self(z)

        log_pz = self.prior.log_prob(z)
        log_pxz = self.emission(recon_x).log_prob(x)

        return log_pz + log_pxz

    def langevin_sample(self, x):
        score = self.grad_with_z(x)
        return self.particle + self.hparams.step_size * score + math.sqrt(2*self.hparams.step_size) * torch.randn(self.particle.shape)

    def grad_with_params(self, x):
        log_likelihood = self.log_likelihood(x, self.particle)
        log_likelihood.backward(retain_graph=False, inputs=self.parameters())
        return torch.stack([param.grad for param in self.parameters()])

    def grad_with_z(self, x):
        log_likelihood = self.log_likelihood(x, self.particle)
        log_likelihood.backward(retain_graph=False, inputs=self.particle)
        return self.particle.grad

    def training_step(self, batch):
        x, y = batch

        opt = self.optimizers()
        opt.zero_grad()

        mc_grad = - self.grad_with_params(x)
        self.particle = self.langevin_sample(x)

        for _ in range(self.hparams.total_step):
            mc_grad -= self.grad_with_params(x)
            self.particle = self.langevin_sample(x)

        mc_grad /= self.hparams.total_step

        with torch.no_grad():
            for param, grad in zip(self.parameters(), mc_grad):
                param.grad = grad

        opt.step()


    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.paramters(), lr=3e-4)








