import argparse
import torch
import matplotlib.pyplot as plt

from gan.model import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an image using GAN")
    parser.add_argument("weights_path", type=str, help="Path to the G weights")
    parser.add_argument(
        "--image_size",
        type=int,
        help="Image size",
        default=32,
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        help="Latent dimension",
        default=100,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output image",
        required=False,
    )
    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def eval(args):
    device = get_device()
    print(f"Using device: {device}")

    device = torch.device(device)
    generator = Generator(
        img_size=args.image_size,
        latent_dim=args.latent_dim,
        img_channels=3,
        feature_maps=64,
    ).to(device)
    generator.load_state_dict(torch.load(args.weights_path, map_location=device))
    generator.eval()

    with torch.no_grad():
        random_input = torch.randn(1, args.latent_dim, 1, 1, device=device)
        generated_image = generator(random_input).cpu().squeeze()

    # Normalize the image to be between 0 and 1
    generated_image = (generated_image + 1) / 2

    # Convert to numpy array and transpose to (H, W, C) for plotting
    np_image = generated_image.numpy().transpose(1, 2, 0)

    # Plot the generated image
    plt.imshow(np_image)
    plt.axis("off")
    plt.show()

    if args.output_path:
        plt.savefig(args.output_path)


def main():
    args = parse_args()
    eval(args)


if __name__ == "__main__":
    main()
