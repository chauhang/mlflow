import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset


class IrisClassification(pl.LightningModule):
    def __init__(self):
        super(IrisClassification, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        a, y_hat = torch.max(logits, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), y.cpu())
        return {"test_loss": loss, "test_acc": torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss, "test_acc": avg_test_acc}
        return {
            "avg_test_loss": avg_loss,
            "avg_test_acc": avg_test_acc,
            "log": logs,
            "progress_bar": logs,
        }


if __name__ == "__main__":
    model = IrisClassification()

    trainer = pl.Trainer(max_epochs=20, gpus=2, distributed_backend="ddp", prepare_data_per_node=False)
    # trainer = pl.Trainer(max_epochs=20)

    iris = load_iris()
    df = iris.data
    target = iris["target"]

    data = torch.Tensor(df).float()
    labels = torch.Tensor(target).long()

    data_set = TensorDataset(data, labels)
    train_set, val_set = random_split(data_set, [130, 20])
    train_set, test_set = random_split(train_set, [110, 20])
    train_loader = DataLoader(dataset=train_set, batch_size=8, num_workers=32)
    test_loader = DataLoader(dataset=test_set, batch_size=8, num_workers=32)
    val_loader = DataLoader(dataset=val_set, batch_size=8, num_workers=32)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)
    torch.save(trainer.model, "iris_gpu.pt")
#    torch.save(trainer.model.state_dict(), "iris_gpu_state_dict")
