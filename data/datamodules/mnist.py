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
    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool):
        super(MNISTDataloader, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

    @property
    def num_train_data(self):
        return len(self.data_train)

    @property
    def num_val_data(self):
        return len(self.data_val)

    def setup(self, stage: Optional[str] = None) -> None:
        trainset = MNISTDataset(
            root='data', train=True, download=True, transform=self.transforms
        )

        total_num_data = len(trainset)
        n_train = int(0.9 * total_num_data)
        n_val = total_num_data - n_train

        self.data_train, self.data_val = random_split(
            trainset, [n_train, n_val]
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
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )