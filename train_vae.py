import os

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
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
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_name', default='mnist', type=str)
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--output_dim', default=1000, type=int)
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

    (train_loader, _), val_loader = get_dataloader(
        name=args.data_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True
    )

    # ------------
    # training
    # ------------

    tb_logger = TensorBoardLogger(
        save_dir='checkpoints', name='VAE', version=f'trial_{args.data_name}'
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, 'model_ckpt'),
        save_top_k=-1,
        monitor='val_recon_loss',
        auto_insert_metric_name=True,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[model_checkpoint],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=False)
    )
    trainer.fit(vae, train_loader, val_loader)


if __name__ == '__main__':
    cli_main()
