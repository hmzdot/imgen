import torch
import torch.optim as optim
import torch.nn as nn
import logging
from tqdm import tqdm
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator

logging.basicConfig(level=logging.INFO)

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


def train_gan(
    generator,
    discriminator,
    loader,
    epochs=100,
    lr=0.0002,
    betas=(0.5, 0.999),
):
    device = get_device()
    generator.to(device)
    discriminator.to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    adversarial_loss = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):
        for real_img, _ in tqdm(loader, desc=f"Epoch #{epoch + 1}"):
            # Move real_img to the same device
            real_img = real_img.to(device)
            valid = torch.ones(real_img.size(0), 1).to(device)
            fake = torch.zeros(real_img.size(0), 1).to(device)

            # Train generator
            optimizer_g.zero_grad()
            z = torch.randn(real_img.size(0), 100, 1, 1).to(device)
            gen_img = generator(z)
            g_loss = adversarial_loss(discriminator(gen_img), valid)
            g_loss.backward()
            optimizer_g.step()

            # Train discriminator
            optimizer_d.zero_grad()
            real_loss = adversarial_loss(discriminator(real_img), valid)
            # Generate new fake images for discriminator
            z = torch.randn(real_img.size(0), 100, 1, 1).to(device)
            gen_img = generator(z)
            fake_loss = adversarial_loss(discriminator(gen_img.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

        logging.info(
            f"Epoch #{epoch + 1}, D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save(generator.state_dict(), f"./snapshots/gw_{timestamp}.pth")
    torch.save(discriminator.state_dict(), f"./snapshots/dw_{timestamp}.pth")


if __name__ == "__main__":
    train_gan(Generator(), Discriminator(), loader)
