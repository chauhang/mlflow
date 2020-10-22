import argparse
import torch
import time
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import pytorch_autolog


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.test_transforms = transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def prepare_data(self):
        trainset = torchvision.datasets.CIFAR10(
            root="./CIFAR10", train=True, download=True, transform=self.train_transforms
        )
        testset = torchvision.datasets.CIFAR10(
            root="./CIFAR10", train=False, download=True, transform=self.test_transforms
        )
        return trainset, testset

    def setup(self, stage=None):
        if stage == "fit" or stage == "None":
            self.dataset, _ = self.prepare_data()
            self.train_set, self.val_set = random_split(self.dataset, [45000, 5000])

        if stage == "test" or stage == "None":
            _, self.test_set = self.prepare_data()

    def train_dataloader(self):

        return DataLoader(self.train_set, batch_size=64, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=64, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=64, shuffle=False)


class AlexNet(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def cross_entropy_loss(self, logits, labels):
        """
        Loss Fn to compute loss
        """
        # labels = labels.squeeze(1)
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

        # prediction_score, pred_label_idx = torch.topk(output, 1)
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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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
        "--mlflow_run_name", help="Name of MLFLOW experiment run in which results would be dumped",
    )
    parser.add_argument(
        "--tracking_uri",
        default="http://localhost:5000",
        help="Address of the MLFLOW tracking uri ",
    )
    args = parser.parse_args()
    tracking_uri = args.tracking_uri
    mlflow.tracking.set_tracking_uri(tracking_uri)
    experiment_name = args.mlflow_experiment_name
    mlflow.set_experiment(experiment_name)
    run_name = args.mlflow_run_name
    mlflow.start_run(run_name=run_name)
    trainer = pl.Trainer(max_epochs=int(args.max_epochs))
    model = AlexNet()
    dm = DataModule()
    dm.setup("fit")
    pytorch_autolog.autolog()
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = round((time.time() - start_time) / 60, 2)
    testloader = dm.setup("test")
    trainer.test(datamodule=testloader)
    model = trainer.model
