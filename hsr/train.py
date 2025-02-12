import glob
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers
from pytorch_lightning.plugins.environments import SLURMEnvironment

from hsr.datasets import create_dataloader, create_dataloaders
from hsr.utils.misc import TQDMProgressBarNoVNum, get_class, suppress_warning


# https://github.com/Lightning-AI/lightning/issues/6389#issuecomment-1377759948
class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


def config_torch():
    pl.seed_everything(42)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    suppress_warning("trimesh")


def config_hydra(cfg):
    cfg.dataset.train.data_dir = Path(cfg.data_root) / cfg.dataset.train.data_dir
    return


@hydra.main(version_base="1.12", config_path="confs", config_name="base")
def main(cfg):
    config_torch()
    config_hydra(cfg)

    OmegaConf.save(cfg, "config.yaml")

    train_dloader, val_dloader, test_dloader = create_dataloaders(cfg.dataset)
    model = get_class(cfg.model.cls)(cfg)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:04d}-{step:06d}-{val/psnr:.2f}",
        auto_insert_metric_name=False,
        # save_on_train_epoch_end=True,
        every_n_epochs=1,
        save_top_k=1,
    )
    progress_bar_callback = TQDMProgressBarNoVNum(refresh_rate=10)
    if cfg.wandb:
        name = f"{cfg.dataset.train.subject}-{cfg.model.name}-{cfg.version}"
        logger = loggers.WandbLogger(project="HSR", name=name, version=name)
    else:
        logger = loggers.TensorBoardLogger(save_dir="tblogger/", name="", version="")
    val_freq = cfg.model.val_steps // len(train_dloader)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, progress_bar_callback],
        max_steps=cfg.model.max_steps,
        check_val_every_n_epoch=val_freq,
        logger=logger,
        log_every_n_steps=10,
        inference_mode=False,
        plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
        num_sanity_val_steps=0,
    )

    if cfg.model.is_continue or cfg.test_only:
        if cfg.ckpt_path is not None:
            ckpt_path = cfg.ckpt_path
        else:
            ckpt_path = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    else:
        ckpt_path = None

    if cfg.test_only:
        trainer.test(model=model, dataloaders=test_dloader, ckpt_path=ckpt_path)
    else:
        trainer.fit(
            model=model,
            train_dataloaders=train_dloader,
            val_dataloaders=val_dloader,
            ckpt_path=ckpt_path,
        )
        ckpt_path = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        trainer.test(dataloaders=test_dloader, ckpt_path=ckpt_path)

    return


if __name__ == "__main__":
    main()
