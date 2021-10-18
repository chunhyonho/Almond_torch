import os
import pickle
import numpy as np

from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class SingleCellDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y

        assert self.X.shape[0] == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return np.log(1+self.X[idx]), self.y[idx], idx


class SingleCellDataloader(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            data_name: str,
            num_train_data: int,
            num_val_data: int
    ):
        super(SingleCellDataloader, self).__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_name = data_name
        self.num_train_data = num_train_data
        self.num_val_data = num_val_data

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(self.data_dir, f"{self.data_name}_train.pkl"), 'rb') as f:
            X, y = pickle.load(f)
        trainset = SingleCellDataset(X, y)

        self.data_train, self.data_val = random_split(
            trainset, [self.num_train_data, self.num_val_data]
        )

        with open(os.path.join(self.data_dir, f"{self.data_name}_test.pkl"), 'rb') as f:
            X, y = pickle.load(f)
        self.data_test = SingleCellDataset(X, y)

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
            batch_size=len(self.data_val)//4,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=len(self.data_test)//4,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True
        )


class ZHAI_DataModule(SingleCellDataloader):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, pin_memory: bool, num_train_data: int, num_val_data: int):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            data_name='zhai',
            num_train_data=num_train_data,
            num_val_data=num_val_data
        )

class PURE_DataModule(SingleCellDataloader):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, pin_memory: bool, num_train_data: int, num_val_data: int):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            data_name='pure',
            num_train_data=num_train_data,
            num_val_data=num_val_data
        )


class PROCESS_DataModule(SingleCellDataloader):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, pin_memory: bool, num_train_data: int, num_val_data: int):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            data_name='process',
            num_train_data=num_train_data,
            num_val_data=num_val_data
        )