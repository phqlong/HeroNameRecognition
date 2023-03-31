from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision import models



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        loss = torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0))
        return loss


class HeroImagesLitModule(LightningModule):
    """Example of LightningModule for HeroImages classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        # net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Save net model to caculate self.parameters() in configure_optimizers
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove the last layer (classifier)

        # loss function
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0)
        # self.criterion = TripletLoss(margin=1.0)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def model_step(self, batch: Any):
        anchor, positive, negative = batch
                
        anchor_embedding = self.forward(anchor)
        positive_embedding = self.forward(positive)
        negative_embedding = self.forward(negative)

        loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        batch_data, targets = batch
        loss = self.model_step(batch_data)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from 
        # all batches of the epoch this may not be an issue when training on HeroImages
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        batch_data, targets = batch
        loss = self.model_step(batch_data)

        # update and log metrics
        val_loss = self.val_loss(loss)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass
    
    def test_step(self, batch: Any, batch_idx: int):
        batch_data, targets = batch
        loss = self.model_step(batch_data)

        # update and log metrics
        test_loss = self.test_loss(loss)
        self.log("test/loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}


    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # Lr and weight_decay are partially initialized in hydra.utils.instantiate(cfg.model)
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "default.yaml")
    _ = hydra.utils.instantiate(cfg)
