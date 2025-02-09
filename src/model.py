import torch.nn as nn
import math


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64, img_size=32):
        """
        # Parameters
        - latent_dim: Dimension for the latent vector
        - img_channels: Number of channels in the image (3 for RGB)
        - feature_maps: Base number of features for convolution
        - img_size: Target image size (must be a power of 2)
        """
        super(Generator, self).__init__()

        # Calculate number of layers needed
        num_layers = int(math.log2(img_size)) - 2  # -2 because we start at 4x4

        layers = []
        # Initial layer: latent_dim -> (feature_maps * 2^num_layers) at 4x4
        initial_features = feature_maps * (2**num_layers)
        layers.extend(
            [
                nn.ConvTranspose2d(latent_dim, initial_features, 4, 1, 0, bias=False),
                nn.BatchNorm2d(initial_features),
                nn.ReLU(True),
            ]
        )

        # Add upsampling layers
        current_size = 4
        in_features = initial_features
        for i in range(num_layers - 1):  # Process all but the last layer in the loop
            out_features = in_features // 2
            layers.extend(
                [
                    nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_features),
                    nn.ReLU(True),
                ]
            )
            current_size *= 2
            in_features = out_features

        # Final layer to get to img_channels (last upsampling + conversion to image)
        layers.extend(
            [
                nn.ConvTranspose2d(in_features, img_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: (batch_size, img_channels, 32, 32) -> (batch_size, feature_maps, 16, 16)
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, feature_maps, 16, 16) -> (batch_size, feature_maps*2, 8, 8)
            nn.Conv2d(
                in_channels=feature_maps,
                out_channels=feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, feature_maps*2, 8, 8) -> (batch_size, feature_maps*4, 4, 4)
            nn.Conv2d(
                in_channels=feature_maps * 2,
                out_channels=feature_maps * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, feature_maps*4, 4, 4) -> (batch_size, 1, 1, 1)
            nn.Conv2d(
                in_channels=feature_maps * 4,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)


class Critic(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64, img_size=32):
        """
        # Parameters
        - img_channels: Number of channels in the image (3 for RGB)
        - feature_maps: Base number of features for convolution
        - img_size: Input image size (must be a power of 2)
        """
        super(Critic, self).__init__()

        if img_size % 2 != 0:
            raise ValueError("Image size must be a power of 2")

        # Calculate number of layers needed
        num_layers = int(math.log2(img_size)) - 2  # -2 because we end at 4x4

        if num_layers <= 0:
            raise ValueError("Image size must be at least 4x4")

        layers = []
        # Initial layer: img_channels -> feature_maps
        layers.extend(
            [
                nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        # Add downsampling layers
        current_size = img_size // 2
        in_features = feature_maps
        for _ in range(num_layers - 1):  # -1 because we already did first layer
            out_features = in_features * 2
            layers.extend(
                [
                    nn.Conv2d(in_features, out_features, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_features),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_size //= 2
            in_features = out_features

        # Final layer to get to 1 channel
        layers.append(nn.Conv2d(in_features, 1, 4, 1, 0, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img).view(-1, 1)
