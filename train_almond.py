import os
import copy
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
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
        '--ckpt_path',
        default='checkpoints/VAE/trial_process/model_ckpt/epoch=4-step=484.ckpt',
        type=str
    )
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--data_name', default='process', type=str)
    parser.add_argument('--step_size', default=0.02, type=float)
    parser.add_argument('--total_step', default=5000, type=int)
    args = parser.parse_args()

    vae = VAE.load_from_checkpoint(checkpoint_path=args.ckpt_path)

    encoder_mu = copy.deepcopy(vae.encoder_mu)
    encoder_logvar = copy.deepcopy(vae.encoder_logvar)
    decoder = copy.deepcopy(vae.decoder)

    del vae

    # ------------
    # data
    # ------------

    (train_loader, num_data_train), val_loader = get_dataloader(
        name=args.data_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True
    )

    # ------------
    # model
    # ------------


    almond = ALMOND(
        encoder_mu=encoder_mu,
        encoder_logvar=encoder_logvar,
        decoder=decoder,
        step_size=args.step_size,
        total_step=args.total_step,
        batch_size=args.batch_size,
        num_data=num_data_train
    )
    # ------------
    # training
    # ------------

    tb_logger = TensorBoardLogger(
        save_dir='checkpoints', name='ALMOND', version=f'trial_{args.data_name}'
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
        logger=tb_logger,
        callbacks=[model_checkpoint],
        plugins=DDPPlugin(find_unused_parameters=False)
    )
    trainer.fit(almond, train_loader, val_loader)


if __name__ == '__main__':
    cli_main()