import argparse

from sklearn.datasets import load_iris
import torch.nn as nn
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
import torch
import pytorch_autolog
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
import mlflow
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


class IRISDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.columns = None

    def prepare_data(self):
        iris = load_iris()
        df = iris.data
        self.columns = iris.feature_names
        target = iris["target"]
        data = torch.Tensor(df).float()
        labels = torch.Tensor(target).long()
        data_set = TensorDataset(data, labels)
        return data_set

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            iris_full = self.prepare_data()
            self.train_set, self.val_set = random_split(iris_full, [130, 20])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.train_set, self.test_set = random_split(self.train_set, [110, 20])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=4)


class IrisClassification(pl.LightningModule):
    def __init__(self):
        super(IrisClassification, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        """
        Loss Fn to compute loss
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        training the data as batches and returns training loss on each batch
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes test accuracy
        """

        x, y = test_batch
        output = self.forward(x)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), y.cpu())
        return {"test_acc": torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        return {"avg_test_acc": avg_test_acc}

    def configure_optimizers(self):
        """
        Creates and returns Optimizer
        """

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        return self.optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_epochs",
        default=3,
        help="Describes the number of times a neural network has to be trained",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        help="Name of MLFLOW experiment in which results would be dumped",
    )
    parser.add_argument(
        "--tracking_uri",
        default="http://localhost:5000/",
        help="Address of the tracking URI",
    )
    args = parser.parse_args()
    mlflow.tracking.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.start_run()
    trainer = pl.Trainer(max_epochs=int(args.max_epochs))
    model = IrisClassification()
    dm = IRISDataModule()
    dm.setup("fit")
    pytorch_autolog.autolog()
    trainer.fit(model, dm)
    testloader = dm.setup("test")
    trainer.test(datamodule=testloader)
