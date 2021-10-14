import pickle
import torch
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split

TRAIN_VAL_RATIO = 0.8
TEST_DATA_SIZE = 10000


class SingleCellDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y

        assert self.X.shape[0] == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


class MNISTDataset(MNIST):
    def __init__(self, root: str, train: bool, download: bool, transform):
        super().__init__(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        return *super().__getitem__(index), index


def get_dataloader(name: str, batch_size: int, num_workers: int = 0, train: bool = True):
    if name in ('zhai', 'pure', 'process'):
        if train:
            with open(f"data/SINGLECELL/{name}_train.pkl", 'rb') as f:
                X, y = pickle.load(f)
            train_ds = SingleCellDataset(X, y)

            with open(f"data/SINGLECELL/{name}_val.pkl", 'rb') as f:
                X, y = pickle.load(f)
            val_ds = SingleCellDataset(X, y)

        else:
            with open(f"data/SINGLECELL/{name}_test.pkl", 'rb') as f:
                X, y = pickle.load(f)
            test_ds = SingleCellDataset(X, y)

    elif name == 'mnist':
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        if train:
            train_ds = MNISTDataset(
                root='data', train=True, download=True, transform=mnist_transform
            )
            train_ds, val_ds = random_split(
                train_ds,
                [int(TRAIN_VAL_RATIO * len(train_ds)), len(train_ds) - int(TRAIN_VAL_RATIO * len(train_ds))]
            )
        else:
            test_ds = MNISTDataset(
                root='data', train=False, download=True, transform=mnist_transform
            )

    if train:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return (train_loader, len(train_ds)), val_loader

    else:

        test_ds, _ = random_split(
            test_ds,
            [TEST_DATA_SIZE, len(test_ds) - TEST_DATA_SIZE]
        )

        test_loader = DataLoader(
            test_ds, batch_size=TEST_DATA_SIZE, shuffle=False
        )

        for i, batch in enumerate(test_loader):
            if i == 0:
                return batch
