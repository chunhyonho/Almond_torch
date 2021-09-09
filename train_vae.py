from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from pytorch_lightning.plugins import DDPPlugin
from src.vae import VAE
from data.utils import *


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------

    parser = ArgumentParser()
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--output_dim', default=784, type=int)
    parser.add_argument('--hidden_dim', nargs='+', default=[], type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    args = parser.parse_args()

    # ------------
    # model
    # ------------

    vae = VAE(
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        encoder='MLP',
        decoder='MLP',
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate
    )

    # ------------
    # data
    # ------------

    train_loader, test_loader = get_dataloader(name='mnist', batch_size=512, num_workers=4)

    # ------------
    # training
    # ------------

    tb_logger = TensorBoardLogger(
        save_dir='checkpoints', name='VAE', version='trial'
    )

    trainer = pl.Trainer(
        checkpoint_callback=True,
        gpus=args.gpus,
        logger=tb_logger,
        max_epochs=args.max_epochs,
        plugins=DDPPlugin(find_unused_parameters=False)
    )
    trainer.fit(vae, train_loader, test_loader)


if __name__ == '__main__':
    cli_main()
