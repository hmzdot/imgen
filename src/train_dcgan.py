import torch
import torch.optim as optim
import torch.nn as nn
import logging
from tqdm import tqdm
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

cifar_10 = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
loader = DataLoader(cifar_10, batch_size=64, shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train(
    G,
    D,
    loader,
    epochs=100,
    lr=0.0002,
    betas=(0.5, 0.999),
):
    """
    Trains a GAN model.
    # Parameters
    - G: Generator model
    - D: Discriminator model
    - loader: DataLoader for the dataset
    - epochs: Number of epochs to train for
    - n_critic: Number of critic updates per generator update
    - lr: Learning rate
    """
    device = get_device()
    G.to(device)
    D.to(device)

    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):
        for real_img, _ in tqdm(loader, desc=f"Epoch #{epoch + 1}"):
            # Move real_img to the same device
            real_img = real_img.to(device)
            valid = torch.ones(real_img.size(0), 1).to(device)
            fake = torch.zeros(real_img.size(0), 1).to(device)

            # Train generator
            optimizer_G.zero_grad()
            z = torch.randn(real_img.size(0), 100, 1, 1).to(device)
            gen_img = G(z)
            g_loss = adversarial_loss(D(gen_img), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(D(real_img), valid)
            # Generate new fake images for discriminator
            z = torch.randn(real_img.size(0), 100, 1, 1).to(device)
            gen_img = G(z)
            fake_loss = adversarial_loss(D(gen_img.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        logging.info(
            f"Epoch #{epoch + 1}, D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save(G.state_dict(), f"./snapshots/gw_{timestamp}.pth")
    torch.save(D.state_dict(), f"./snapshots/dw_{timestamp}.pth")


if __name__ == "__main__":
    train(Generator(), Discriminator(), loader)
