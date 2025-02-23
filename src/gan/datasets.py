import torchvision.datasets as datasets
import torchvision.transforms as transforms


CIFAR_10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_10_STD = (0.2470, 0.2435, 0.2616)
CIFAR_10_CHANNELS = 3
CIFAR_10_SIZE = 32


def CIFAR_10(root="./data", train=True):
    return datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD),
            ]
        ),
    )


CELEBA_CHANNELS = 3
CELEBA_SIZE = 128


def CELEBA(root="./data", split="train"):
    return datasets.CelebA(
        root=root,
        split=split,
        download=True,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
