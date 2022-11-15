from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, args, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.args = args

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path,
                         download=True)

        # Note: Network trains about 2-3x faster if we don't
        # resize (keeping the orig. 32x32 res.)
        # However, the test set accuracy is approx 10% lower
        # on the original 32x32 resolution
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((70, 70)),
            torchvision.transforms.RandomCrop((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((70, 70)),
            torchvision.transforms.CenterCrop((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return

    def setup(self, stage=None):
        train = datasets.CIFAR10(root=self.data_path,
                                 train=True,
                                 transform=self.train_transform,
                                 download=True)

        self.test = datasets.CIFAR10(root=self.data_path,
                                     train=False,
                                     transform=self.test_transform,
                                     download=True)

        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train,
                                  batch_size=self.args.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=self.args.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid,
                                  batch_size=self.args.batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=self.args.num_workers)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test,
                                 batch_size=self.args.batch_size,
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=self.args.num_workers)
        return test_loader
