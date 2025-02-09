# Image Generation - Small

Image generation using GAN, with a small dataset (CIFAR-10)

## Implemented Variations

- [X] DCGAN (Deep Convolutional GAN)
- [X] WGAN-WC (Wasserstein GAN with Weight Clipping)
- [X] WGAN-GP (Wassertstein GAN with Gradient Clipping)

## Build and Run

With `uv` installed

```bash
# Clone the repo
git clone git@github.com:hmzdot/imgen-small.git
cd imgen-small

# Install dependencies
uv sync

# Run training
# Under the hood it calls `uv run python -m bin.train`
# (This generates snapshots/dw_{timestamp}.pth and snapshots/gw_{timestamp}.pth)
uv run train --model=dcgan --dataset=cifar10
uv run train --model=wgan_wc --dataset=cifar10
uv run train --model=wgan_gp --dataset=celeba
uv run train --help

# Generate a random image
# Under the hood it calls `uv run python -m bin.eval`
# Takes generator network's weights as the input
uv run eval snapshosts/gw_{timestamp}.pth
uv run eval --help
```
