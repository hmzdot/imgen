"""
Combines DCGAN, WGAN-GP, and WGAN-WC training scripts into a single script.
"""

import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader

from gan.datasets import (
    CIFAR_10,
    CIFAR_10_CHANNELS,
    CIFAR_10_SIZE,
    CELEBA,
    CELEBA_CHANNELS,
    CELEBA_SIZE,
)
from gan.dcgan import train as train_dcgan
from gan.wgan_gp import train as train_wgan_gp
from gan.wgan_wc import train as train_wgan_wc
from gan.model import Generator, Discriminator, Critic


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create data directory
os.makedirs("./data", exist_ok=True)
os.makedirs("./snapshots", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to use, either 'cifar10' or 'celeba'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dcgan",
        help="Model to use, either 'dcgan', 'wgan_gp', or 'wgan_wc'",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--clip_value", type=float, default=0.01)
    parser.add_argument("--lambda_gp", type=float, default=10)
    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def train(args):
    dataset = CIFAR_10() if args.dataset == "cifar10" else CELEBA()
    img_size = CIFAR_10_SIZE if args.dataset == "cifar10" else CELEBA_SIZE
    img_channels = CIFAR_10_CHANNELS if args.dataset == "cifar10" else CELEBA_CHANNELS
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = get_device()

    logging.info(f"Using dataset: {args.dataset}")
    logging.info(f"Using model: {args.model}")
    logging.info(f"Using device: {device}")

    if args.model == "dcgan":
        G = Generator(
            latent_dim=args.latent_dim,
            img_channels=img_channels,
            img_size=img_size,
        )
        D = Discriminator(
            img_channels=img_channels,
            img_size=img_size,
        )
        train_dcgan(
            G,
            D,
            loader,
            epochs=args.epochs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            device=device,
        )
    elif args.model == "wgan_gp":
        G = Generator(
            latent_dim=args.latent_dim,
            img_channels=img_channels,
            img_size=img_size,
        )
        D = Critic(
            img_channels=img_channels,
            img_size=img_size,
        )
        train_wgan_gp(
            G,
            D,
            loader,
            epochs=args.epochs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            factor=args.lambda_gp,
            device=device,
        )
    elif args.model == "wgan_wc":
        G = Generator(
            latent_dim=args.latent_dim,
            img_channels=args.img_channels,
            img_size=args.img_size,
        )
        D = Critic(img_channels=args.img_channels, img_size=args.img_size)
        train_wgan_wc(
            G,
            D,
            loader,
            epochs=args.epochs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            device=device,
        )
    else:
        raise ValueError(f"Invalid model: {args.model}")


def main():
    args = parse_args()
    if args.verbose:
        for key, value in args._get_kwargs():
            logging.info(f"{key}: {value}")
    train(args)


if __name__ == "__main__":
    main()
