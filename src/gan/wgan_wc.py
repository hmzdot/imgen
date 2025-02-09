import torch
import torch.optim as optim
import torch.nn as nn
import logging
import itertools
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Literal

from .model import Generator, Critic
from .datasets import CIFAR_10
from .lipschitz import weight_clipping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def critic_loss(D, real_samples, fake_samples):
    # For critic we want to maximize D(real) - D(fake)
    # When using an optimizer that minimizes, we define:
    loss = torch.mean(D(fake_samples)) - torch.mean(D(real_samples))
    return loss


def generator_loss(D, fake_samples):
    # For generator we want to maximize D(fake), which is equivalent to minimizing -D(fake)
    loss = -torch.mean(D(fake_samples))
    return loss


def train(
    G,
    D,
    loader,
    epochs: Union[int, Literal["inf"]] = 100,
    n_critic=5,
    lr=0.00005,
    device="cpu",
):
    """
    Trains a GAN model.
    # Parameters
    - G: Generator model
    - D: Discriminator model
    - loader: DataLoader for the dataset
    - epochs: Number of epochs to train for, or "inf" to run indefinitely
    - n_critic: Number of critic updates per generator update
    - lr: Learning rate
    """
    # Initialize TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    writer = SummaryWriter(f"runs/wgan_wc_{timestamp}")

    device = torch.device(device)
    G.to(device)
    D.to(device)

    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)

    optimizer_G = optim.RMSprop(G.parameters(), lr=lr)
    optimizer_D = optim.RMSprop(D.parameters(), lr=lr)

    if epochs == "inf":
        epochs_range = itertools.count()
    else:
        epochs_range = range(epochs)

    writer_step = 0
    for epoch in epochs_range:
        for real_img, _ in tqdm(loader, desc=f"Epoch #{epoch + 1}"):
            # Move real_img to the same device
            real_img = real_img.to(device)
            batch_size = real_img.size(0)

            # Train discriminator / critic
            # For every G step, we train the D n_critic times
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                z = torch.randn(batch_size, 100, 1, 1, device=device)
                gen_img = G(z).detach()

                loss_D = critic_loss(D, real_img, gen_img)
                loss_D.backward()
                optimizer_D.step()

                # Clip weights to enforce Lipschitz constraint
                weight_clipping(D)

            # Train generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_img = G(z)
            loss_G = generator_loss(D, fake_img)
            loss_G.backward()
            optimizer_G.step()

            # Log losses to TensorBoard
            writer.add_scalar("Loss/Critic", loss_D.item(), writer_step)
            writer.add_scalar("Loss/Generator", loss_G.item(), writer_step)
            writer_step += 1

        if epoch % 10 == 0:  # Log images every 10 epochs
            # Log sample images
            with torch.no_grad():
                sample_z = torch.randn(8, 100, 1, 1, device=device)
                sample_images = G(sample_z)
                # Rescale images from [-1, 1] to [0, 1]
                sample_images = (sample_images + 1) / 2
                writer.add_images("Generated Images", sample_images, epoch)

            # Save models
            torch.save(G.state_dict(), f"./snapshots/gw_{timestamp}.pth")
            torch.save(D.state_dict(), f"./snapshots/dw_{timestamp}.pth")

            logging.info(
                f"Saved models to snapshots/gw_{timestamp}.pth and snapshots/dw_{timestamp}.pth"
            )

        logging.info(
            f"Epoch #{epoch + 1}, D loss: {loss_D.item():.4f}, G loss: {loss_G.item():.4f}"
        )

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    loader = DataLoader(CIFAR_10, batch_size=64, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        Generator(img_channels=3, img_size=32),
        Critic(img_channels=3, img_size=32),
        loader,
        device=device,
    )
