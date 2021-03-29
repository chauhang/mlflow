import torch
from argparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import dataset_voc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VOCDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.pin_memory = True
        self.ignore_label = 250
        self.batch_size = 16
        self.num_workers = 3
        self.args = kwargs
        self.train_set = None
        self.val_set = None

    def prepare_data(self):
        """

        Implementation of abstract class
        """

    def setup(self, stage=None):
        self.train_set = dataset_voc.VOC(True)
        self.val_set = dataset_voc.VOC(False)
        print("Len of train_set", len(self.train_set))
        print("Len of val_set", len(self.val_set))
        return self.train_set, self.val_set

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=4,
            metavar="N",
            help="input batch size for training (default: 16)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 0)",
        )
        parser.add_argument('--random_sample', action='store_true', 
            default=True, help='whether to sample the dataset with random sampler')

        return parser

    def train_dataloader(self):
        train_data_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_data_loader

    def val_dataloader(self):
        val_data_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_data_loader
