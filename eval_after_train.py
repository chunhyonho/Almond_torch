from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from src import VAE, ALMOND
from data.utils import get_dataloader


def cli_main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------

    parser = ArgumentParser()
    parser.add_argument(
        '--ckpt_path',
        default='checkpoints/VAE/trial/checkpoints/epoch=4-step=294.ckpt',
        type=str
    )
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--model', default='VAE', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--data_name', default='mnist', type=str)
    parser.add_argument('--step_size', default=0.02, type=float)
    parser.add_argument('--total_step', default=10000, type=int)
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--output_dim', default=784, type=int)
    parser.add_argument('--hidden_dim', nargs='+', default=[], type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()

    # ------------
    # model
    # ------------

    model = VAE(
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        encoder='MLP',
        decoder='MLP',
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate
    )

    if args.model == 'ALMOND':
        model = ALMOND(
            encoder_mu=model.encoder_mu,
            encoder_logvar=model.encoder_logvar,
            decoder=model.decoder,
            step_size=args.step_size,
            total_step=args.total_step,
            batch_size=args.batch_size
        )

    # ------------
    # data
    # ------------

    X, y = get_dataloader(
        name=args.data_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False
    )

    # ------------
    # test
    # ------------

    ### Reconstruction error
    if args.model == 'VAE':
        z, x_hat, _, _ = model._run_step(X)

    elif args.model == 'ALMOND':
        z_init = model.get_init_particle(X)


if __name__ == '__main__':
    cli_main()
