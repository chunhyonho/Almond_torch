import torch
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def get_dataloader(name: str, batch_size: int, num_workers: int = 0):
    if name == 'single_zhai':
        pass
    elif name == 'single_pure':
        pass
    elif name == 'single_process':
        pass
    elif name == 'mnist':
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        train_ds = MNIST(
            root='data', train=True, download=True, transform=mnist_transform
        )

        test_ds = MNIST(
            root='data', train=False, download=True, transform=mnist_transform
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
