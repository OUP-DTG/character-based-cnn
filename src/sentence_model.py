import torch
import torch.nn as nn
from torch.nn.functional import pad


class SimpleModel(nn.Module):
    def __init__(self, args, number_of_classes):
        super().__init__()
        vocab_size = args.number_of_characters + len(args.extra_characters)
        self.fc1 = nn.Linear(vocab_size*args.max_length, 1024)
        self.fc2 = nn.Linear(1024, number_of_classes)
        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x.shape)
        return x

def get_conv_layer(vocab_size: int, kernel_size: int):
        return nn.Sequential(
            nn.Conv1d(
                vocab_size,
                128,
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

        # One convolution layer for each n-gram size.
        vocab_size = args.number_of_characters + len(args.extra_characters)
        self.conv2 = get_conv_layer(vocab_size, 2)
        self.conv3 = get_conv_layer(vocab_size, 3)
        self.conv4 = get_conv_layer(vocab_size, 4)
        self.conv5 = get_conv_layer(vocab_size, 5)
        self.conv6 = get_conv_layer(vocab_size, 6)
        self.conv_layers_to_use = 3

        self.max_conv_length = (args.max_length - 1) // 3

        # Linear / fully-connected layers reduce dimension to match number of classes.
        self.fc1 = nn.Sequential(
            nn.Linear(128 * self.conv_layers_to_use * self.max_conv_length, 1024), nn.ReLU(), nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(1024, number_of_classes)

        # initialize weights
        self._create_weights()

    # utility private functions
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, x):
        x = x.transpose(1, 2)
        # Run different convolutions on input.
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        #x5 = self.conv5(x)
        #x6 = self.conv6(x)
        # Pad and concatenate the results.
        results = [x2, x3, x4] # , x5, x6]
        max_length = max(r.size(2) for r in results)
        padded_results = [pad(r, (0, max_length - r.size(2))) for r in results]
        xout = torch.cat(padded_results, 1)
        #print(f"xout {xout.size()}")
        xout = xout.view(xout.size(0), -1)
        #x = x.view(x.size(0), -1)
        x = self.fc1(xout)
        x = self.fc2(x)
        #print(f"final {x.size()}")
        #print(x.shape)
        return x
