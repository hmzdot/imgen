# Image Generation - Small

Image generation using GAN, with a small dataset (CIFAR-10)

## Implemented Variations

- [X] DCGAN (Deep Convolutional GAN)
- [X] WGAN-WC (Wasserstein GAN with Weight Clipping)
- [ ] WGAN-GP (Wassertstein GAN with Gradient Clipping)

## Build and Run

With `uv` installed

```bash
# Clone the repo
git clone git@github.com:hmzdot/imgen-small.git
cd imgen-small

# Install dependencies
uv sync

# Run training
# (This generates snapshots/dw_{timestamp}.pth and snapshots/gw_{timestamp}.pth)
uv run src/train_dcgan.py       # DCGAN
uv run src/train_wgan_wc.py     # WGAN-WC

# Generate a random image
# Takes generator network's weights as the input
uv run src/eval.py snapshosts/gw_{timestamp}.pth
```

## TODO

- [ ] Implement WGAN-GP
- [ ] Instead of RMSprop, use Adam with (β1=0, β2=0.9)
- [ ] Remove batch normalization in WGAN implementation
