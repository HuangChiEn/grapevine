from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

class Dummy_tch_ds:

    def __init__(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        mnist_full = MNIST(self.data_dir, train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        
    def get_all_dataloader(self):
        tra_ld, val_ld = DataLoader(self.mnist_train), DataLoader(self.mnist_val)
        tst_ld = DataLoader(self.mnist_test)
        return tra_ld, val_ld, tst_ld
