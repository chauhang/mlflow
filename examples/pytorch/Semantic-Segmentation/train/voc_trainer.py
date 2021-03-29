from argparse import ArgumentParser
import torch
import mlflow.pytorch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data_module_voc import VOCDataModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VocSemSegment(pl.LightningModule):

    voc_classes = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    def __init__(self, **kwargs):

        super().__init__()
        self.model = models.segmentation.fcn_resnet101(pretrained=True).to(device)
        self.classes_names = self.voc_classes
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[4] = nn.Conv2d(
            512, len(self.classes_names), kernel_size=(1, 1), stride=(1, 1)
        )
        self.lr = 0.001

    def forward(self, x):

        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)",
        )
        return parser

    def training_step(self, batch):
        img, mask = batch
        img = img.float().to(device)
        mask = mask.long().to(device)
        output = self.forward(img)
        loss_1 = F.cross_entropy(output["out"], mask, ignore_index=250)
        loss_2 = F.cross_entropy(output["aux"], mask, ignore_index=250)
        train_loss = loss_1 + 0.4 * loss_2
        self.log("loss", train_loss, on_step=True, on_epoch=True)
        return {"loss": train_loss}

    def validation_step(self, batch):
        img, mask = batch
        img = img.float().to(device)
        mask = mask.long().to(device)
        output = self(img)
        loss_1 = F.cross_entropy(output["out"], mask, ignore_index=250)
        loss_2 = F.cross_entropy(output["aux"], mask, ignore_index=250)
        loss_val = loss_1 + 0.4 * loss_2
        self.log("val_loss", loss_val, prog_bar=True, logger=True)
        return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss_epoch", loss_val)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=3, min_lr=1e-6, verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    pl.seed_everything(1234)
    parser = ArgumentParser(description="Semantic Segmentation")
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = VocSemSegment.add_model_specific_args(parent_parser=parser)
    parser = VOCDataModule.add_model_specific_args(parent_parser=parser)
    args = parser.parse_args()
    dict_args = vars(args)

    mlflow.pytorch.autolog()

    dm = VOCDataModule(**dict_args)
    dm.setup(stage="fit")

    model = VocSemSegment(**dict_args)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="models", save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping, checkpoint_callback], checkpoint_callback=True
    )
    trainer.fit(
        model, dm,
    )
    if trainer.global_rank == 0:
        with mlflow.start_run(run_name=" Semantic Segmentation"):
            mlflow.pytorch.save_state_dict(trainer.get_model().state_dict(), "models/")
