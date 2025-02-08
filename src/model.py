import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        """
        # Parameters
        - latent_dim: Dimension for the latent vector, the noise used to generate
            the image
        - img_channels: Number of channels in the image, 3 for RGB
        - feature_maps: Number of features for convolution
        """
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # Input: (batch_size, latent_dim, 1, 1) -> (batch_size, feature_maps*8, 4, 4)
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=feature_maps * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # (batch_size, feature_maps*8, 4, 4) -> (batch_size, feature_maps*4, 8, 8)
            nn.ConvTranspose2d(
                in_channels=feature_maps * 8,
                out_channels=feature_maps * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # (batch_size, feature_maps*4, 8, 8) -> (batch_size, feature_maps*2, 16, 16)
            nn.ConvTranspose2d(
                in_channels=feature_maps * 4,
                out_channels=feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # (batch_size, feature_maps*2, 16, 16) -> (batch_size, img_channels, 32, 32)
            nn.ConvTranspose2d(
                in_channels=feature_maps * 2,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

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
    def __init__(self, img_channels=3, feature_maps=64):
        super(Critic, self).__init__()
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
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)
