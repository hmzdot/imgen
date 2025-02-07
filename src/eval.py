import argparse
import torch
import matplotlib.pyplot as plt
from model import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an image using GAN")
    parser.add_argument("weights_path", type=str, help="Path to the generator weights")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device()

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.weights_path, map_location=device))
    generator.eval()

    # Generate a random latent vector
    latent_dim = 100
    with torch.no_grad():
        random_input = torch.randn(1, latent_dim, 1, 1, device=device)
        generated_image = generator(random_input).cpu().squeeze()

    generated_image = (generated_image + 1) / 2

    # Convert to numpy array and transpose to (H, W, C) for plotting
    np_image = generated_image.numpy().transpose(1, 2, 0)

    # Plot the generated image
    plt.imshow(np_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
