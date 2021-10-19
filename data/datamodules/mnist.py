import torch
import torchvision.transforms as transforms
from typing import Optional
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule


class MNISTDataset(MNIST):
    def __init__(self, root: str, train: bool, download: bool, transform):
        super().__init__(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        return *super().__getitem__(index), index


class MNISTDataloader(LightningDataModule):
    def __init__(
            self,
            data_dir:str,
            batch_size: int,
            val_batch_size: int,
            num_workers: int,
            pin_memory: bool,
            num_train_data: int,
            num_val_data: int
    ):
        super(MNISTDataloader, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        self.num_train_data = num_train_data
        self.num_val_data = num_val_data

    def setup(self, stage: Optional[str] = None) -> None:
        trainset = MNISTDataset(
            root=self.data_dir, train=True, download=True, transform=self.transforms
        )

        self.data_train, self.data_val = random_split(
            trainset, [self.num_train_data, self.num_val_data]
        )

        self.data_test = MNISTDataset(
            root=self.data_dir, train=False, download=True, transform=self.transforms
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=len(self.data_test) // 4,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True
        )
