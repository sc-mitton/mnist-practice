import torch.nn as nn

IMG_SIZE = 28


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Apply pooling after the second convolution
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Apply pooling after the second convolution in this block
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.mlp = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten the output of convolutions
        x = self.mlp(x)
        return x

# Next steps:
# - data augmentation ✅
# - make the model more complex
#   - skip connections
#   - more convolutional layers
#   - normalization
