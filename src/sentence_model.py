"""Models for processing input sentences.

Notation:
    B: batch size
    V: vocabulary size
    L: max sentence length
    n: number of parallel convolution layers, representing n different n-grams.
    C: number of output channels, often a parameter to tweak
    N: number of classes i.e. number of dialects
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad


class SimpleModel(nn.Module):
    """A very simple model with just two fully-connected layers.

    We would expect any more sophisticated model to perform better than this.
    """
    def __init__(self, args, number_of_classes):
        super().__init__()
        vocab_size = args.number_of_characters + len(args.extra_characters)
        # First layer has 1024 output channels.
        self.fc1 = nn.Linear(vocab_size*args.max_length, 1024)
        # Second layer reduces channels to the number of classes.
        self.fc2 = nn.Linear(1024, number_of_classes)
        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)
    
    def forward(self, x):
        # x is a tensor of shape (B, V, L)
        x = x.view(x.size(0), -1)  # (B, V*L)
        x = self.fc1(x)  # (B, 1024)
        x = self.fc2(x)  # (B, N)
        return x


def get_conv_layer(vocab_size: int, kernel_size: int, output_channels):
    """Construct a 1D convolution layer.

    Input shape: (B, V, L)
    Output shape: (B, C, L // 3)
    """
    return nn.Sequential(
        nn.Conv1d(
            vocab_size,
            output_channels,
            kernel_size=kernel_size,
            padding=0,
            stride=1,
        ),
        nn.ReLU(),
        nn.MaxPool1d(3),
    )


class SentenceCNN(nn.Module):
    def __init__(self, args, number_of_classes):
        super().__init__()

        convolution_channels = 128  # i.e. C = 128
        # I see no better results from using all 5, and it is slower.
        self.conv_layers_to_use = 3

        # One convolution layer for each n-gram size.
        vocab_size = args.number_of_characters + len(args.extra_characters)
        self.conv_layers = nn.ModuleList(
            [get_conv_layer(vocab_size, i, convolution_channels) for i in (2, 3, 4, 5, 6)]
        )

        # Length reduced by a factor of 3 by MaxPool1d layer.
        self.max_conv_length = (args.max_length - 1) // 3
        convolution_output_size = convolution_channels * self.conv_layers_to_use
        fc_input_size = convolution_output_size * self.max_conv_length

        # Linear / fully-connected layers reduce dimension to match number of classes.
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_size, 1024), nn.ReLU(), nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(1024, number_of_classes)

        # initialize weights
        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, x):
        # x is a tensor of shape (B, L, V)
        x = x.transpose(1, 2)  # (B, V, L)
        # Run different convolutions on input.
        xs = [conv(x) for conv in self.conv_layers[:self.conv_layers_to_use]]
        # Now several tensors of shape (B, C, L // 3).
        # Pad and concatenate the results.
        max_length = max(r.size(2) for r in xs[:self.conv_layers_to_use])
        padded_results = [pad(r, (0, max_length - r.size(2))) for r in xs[:self.conv_layers_to_use]]
        xout = torch.cat(padded_results, 1)  # (B, C * n, L // 3)
        xout = xout.view(xout.size(0), -1)  # (B, C * n * (L // 3))
        x = self.fc1(xout)  # (B, 1024)
        x = self.fc2(x)  # (B, N)
        return x
