# Image Generation - Small

Image generation using GAN, with a small dataset (CIFAR-10)

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
uv run src/train.py

# Generate a random image
# Takes generator network's weights as the input
uv run src/eval.py snapshosts/gw_{timestamp}.pth
```

## TODO

- [ ] Use Wasserstein distance
