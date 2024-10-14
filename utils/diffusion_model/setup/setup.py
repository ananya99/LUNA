import glob
import os
import pathlib
import pickle

import omegaconf
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from datasets.data_module import DataModule, Infos
from diffusion_model import FullDenoisingDiffusion
from utils.data.abstract_datatype import AbstractDataModule, AbstractDatasetInfos
from utils.data.misc import setup_wandb

def get_resume(
    cfg: omegaconf.DictConfig,
    dataset_infos: AbstractDatasetInfos,
    checkpoint_path: pathlib.Path,
) -> tuple:
    """
    Loads a model from a checkpoint and updates the experiment configuration.

    Parameters:
    cfg: Configuration object containing settings and parameters for the experiment.
    dataset_infos: Dataset-specific information required for model initialization.
    checkpoint_path: The file path to the saved model checkpoint.
    test: A boolean flag indicating if the model is being loaded for testing (True) or resuming training (False).

    Returns:
    Tuple containing the updated configuration object and the loaded model.
    """
    # Load the model from the specified checkpoint
    model = FullDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path, dataset_infos=dataset_infos, cfg=cfg
    )

    # Return the updated configuration and the loaded model
    return cfg, model

def setup_dataset(cfg: omegaconf.DictConfig, dataset_path: pathlib.Path = None,
                  dataset_infos_path: pathlib.Path = None, load_from_disk: bool = False) -> tuple:
    """Set up the dataset based on the configuration provided."""
    datamodule = DataModule(cfg)
    dataset_infos = Infos(datamodule, cfg)
    return datamodule, dataset_infos


def setup_model(
    cfg: omegaconf.DictConfig,
    dataset_infos: AbstractDatasetInfos,
    checkpoint_path: pathlib.Path = None,
) -> FullDenoisingDiffusion:
    """
    Set up the model based on the configuration and dataset information.

    Parameters:
    cfg: Configuration object containing model settings and other parameters.
    dataset_infos: Information specific to the dataset.

    Returns:
    model: Initialized model based on the provided configuration.
    """
    if cfg.general.mode == "train_and_test":
        pass
    else:
        # Set up for testing only
        cfg, _ = get_resume(cfg, dataset_infos, checkpoint_path)

    # Initialize the model
    model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos)

    return model


def create_model_checkpoint_callbacks(cfg: omegaconf.DictConfig) -> list:
    """Create model checkpoint callbacks based on configuration."""
    save_model = cfg.validation.save_model if cfg.validation.save_model is not None else True
    save_top_k = cfg.validation.save_top_k_models if cfg.validation.save_top_k_models is not None else 20
    monitor_metric = cfg.validation.check_val_monitor if cfg.validation.check_val_monitor else "val_loss/position_mse"
    check_every_n_epochs = cfg.validation.check_val_every_n_epochs if cfg.validation.check_val_every_n_epochs is not None else 20

    callbacks = []
    if save_model:
        callbacks.append(ModelCheckpoint(dirpath="checkpoints", filename="{epoch}",
                                         monitor=monitor_metric, save_top_k=save_top_k,
                                         mode="min", every_n_epochs=check_every_n_epochs))
        callbacks.append(ModelCheckpoint(dirpath="checkpoints", filename="last_epoch", every_n_epochs=1))

    return callbacks


def create_lr_monitor_callback() -> LearningRateMonitor:
    """Create a learning rate monitor callback."""
    return LearningRateMonitor(logging_interval="epoch")


def create_early_stopping_callback(cfg: omegaconf.DictConfig, monitor_metric: str) -> EarlyStopping:
    """Create an early stopping callback based on configuration."""
    return EarlyStopping(monitor=monitor_metric, patience=cfg.validation.early_stopping_patience, mode="min")


def setup_callbacks(cfg: omegaconf.DictConfig, datamodule: AbstractDataModule) -> list:
    """Set up training callbacks based on the configuration."""
    callbacks = create_model_checkpoint_callbacks(cfg)
    callbacks.append(create_lr_monitor_callback())

    if cfg.validation.early_stopping:
        monitor_metric = cfg.validation.check_val_monitor if cfg.validation.check_val_monitor else "val_loss/position_mse"
        callbacks.append(create_early_stopping_callback(cfg, monitor_metric))

    return callbacks


def setup_trainer(cfg: omegaconf.DictConfig, callbacks: list) -> Trainer:
    """Set up the PyTorch Lightning Trainer based on the configuration and callbacks."""
    fast_dev_run = cfg.train.fast_dev_run if cfg.train.fast_dev_run is not None else False
    if fast_dev_run:
        print("[WARNING]: The model will run with fast_dev_run.")

    gpus = cfg.distribute.gpus_per_node if cfg.distribute.gpus_per_node is not None else 1
    max_epochs = cfg.train.n_epochs if cfg.train.n_epochs is not None else 5000
    check_val_every_n_epochs = cfg.validation.check_val_every_n_epochs if cfg.validation.check_val_every_n_epochs is not None else 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if wandb.run and local_rank == 0:
        setup_wandb(cfg)

    return Trainer(
        devices=gpus,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epochs,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
        log_every_n_steps=50 if fast_dev_run else 1,
        enable_progress_bar=cfg.general.enable_progress_bar,
    )


