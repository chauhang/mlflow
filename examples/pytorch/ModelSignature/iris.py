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
    from tests.pytorch.iris import IrisClassification
    from tests.pytorch.iris_data_module import IrisDataModule

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    mlflow.start_run()

    model = IrisClassification(**dict_args)
    dm = IrisDataModule()

    dm.setup("fit")
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    testloader = dm.setup("test")
    trainer.test(datamodule=testloader)
    test_accuracy = trainer.callback_metrics.get("test_acc")
    mlflow.log_metric("Avg_Test_Acc", float(test_accuracy))
    input_df, output_df = extracting_model_input_output_dataframes(trainer)
    signature = infer_signature(input_df, output_df)
    mlflow.pytorch.log_model(trainer.model, "models", signature=signature, input_example=input_df)
