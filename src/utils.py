import torch

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

    elif distribution == 'Poisson':
        def rv(rate: torch.Tensor):
            return torch.distributions.Poisson(
                rate
            )

        return rv