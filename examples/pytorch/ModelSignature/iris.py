import argparse
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from pytorch_lightning.metrics.functional import accuracy
from sklearn.datasets import load_iris
from torch.nn import functional as F
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

input_df = None
output_df = None


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
        self.log("train_loss", loss)
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
        self.log("val_loss", avg_loss)

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes test accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        self.log("test_loss", loss)
        self.log("test_acc", test_acc)
        return {"test_loss": loss, "test_acc": test_acc}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        """
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss)
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Creates and returns Optimizer
        """
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        return self.optimizer


def extracting_model_input_output_dataframes(trainer):
    """
    the function will return model input and model output as dataframes
    """
    input_data = None
    input_subset = None
    input_sample = None
    signature = None
    output = None
    batch_size = 5
    model_input = None
    model_output = None
    input_data = trainer.datamodule.train_dataloader().dataset
    sample_indices = []
    for i in range(batch_size):
        sample_indices.append(i)
    input_subset = DataLoader(input_data, batch_size=5, sampler=SubsetRandomSampler(sample_indices))
    for data in input_subset:
        x1, y1 = data
        if x1.dim() == 4:
            a, x, y, z = x1.shape
            model_input = x1.reshape(a, y * z)
        if x1.dim() == 2:
            model_input = x1
        pred1 = trainer.model.forward(x1)
        actual_value1, model_output = torch.max(pred1, dim=1)
        model_output = torch.transpose(model_output, -1, 0)
    if trainer.datamodule.columns == None:
        trainer.datamodule.columns = range(model_input.shape[1])
    input_df = pd.DataFrame(model_input.numpy(), columns=trainer.datamodule.columns)
    output_df = pd.DataFrame(model_output.numpy())
    return input_df, output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=3,
        help="Describes the number of times a neural network has to be trained",
    )
    args = parser.parse_args()
    mlflow.start_run()
    trainer = pl.Trainer(max_epochs=int(args.max_epochs))
    model = IrisClassification()
    dm = IRISDataModule()
    dm.setup("fit")
    trainer.fit(model, dm)
    testloader = dm.setup("test")
    trainer.test(datamodule=testloader)
    test_accuracy = trainer.callback_metrics.get("avg_test_acc")
    mlflow.log_metric("Avg_Test_Acc", float(test_accuracy))
    input_df, output_df = extracting_model_input_output_dataframes(trainer)
    signature = infer_signature(input_df, output_df)
    mlflow.pytorch.log_model(trainer.model, "models", signature=signature, input_example=input_df)
