from typing import Any, Dict, Optional, Tuple
import glob

import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

from src.datamodules.components.dataset import HeroImagesDataset


class HeroImagesDataModule(LightningDataModule):
    """Example of LightningDataModule for HeroImages dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/hero_images/",
        train_dataset_path: str = "data/train_dataset.pt",
        val_dataset_path: str = "data/val_dataset.pt",
        # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 16,
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        if not os.path.exists(self.hparams.train_dataset_path) and not os.path.exists(self.hparams.val_dataset_path):
            hero_images_path_list = glob.glob(self.hparams.data_dir+"*")

            # Get labels for hero images
            hero_images_path_2_label = {path: path.split("\\")[-1].split(".")[0] for path in hero_images_path_list}

            train_images, val_images = train_test_split(hero_images_path_list, test_size=0.2, random_state=42)

            # Create the datasets
            data_train = HeroImagesDataset(train_images, hero_images_path_2_label, num_triplets=20, train=True, to_tensor=True)
            date_val = HeroImagesDataset(val_images, hero_images_path_2_label, num_triplets=10, train=True, to_tensor=True)
            
            # Save datasets
            torch.save(data_train, self.hparams.train_dataset_path)
            torch.save(date_val, self.hparams.val_dataset_path)


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = torch.load(self.hparams.train_dataset_path)
            self.data_val = torch.load(self.hparams.val_dataset_path)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "default.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
