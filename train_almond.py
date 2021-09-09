from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from pytorch_lightning.plugins import DDPPlugin
from src.vae import VAE
from data.utils import *
from src.almond import ALMOND


def cli_main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------

    parser = ArgumentParser()
    parser.add_argument(
        '--checkpoint_path',
        default='checkpoints/VAE/trial/checkpoints/epoch=4-step=294.ckpt',
        type=str
    )
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--step_size', default=0.02, type=float)
    parser.add_argument('--total_step', default=10000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    args = parser.parse_args()

    vae = VAE.load_from_checkpoint(checkpoint_path=args.checkpoint_path)

    # ------------
    # model
    # ------------

    almond = ALMOND(
        encoder_mu=vae.encoder_mu,
        encoder_logvar=vae.encoder_logvar,
        decoder=vae.decoder,
        step_size=args.step_size,
        total_step=args.total_step,
        batch_size=args.batch_size
    )

    # ------------
    # data
    # ------------

    train_loader, test_loader = get_dataloader(name='mnist', batch_size=args.batch_size, num_workers=4)

    # ------------
    # training
    # ------------

    tb_logger = TensorBoardLogger(
        save_dir='checkpoints', name='ALMOND', version='trial'
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