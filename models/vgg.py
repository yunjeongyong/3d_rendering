import torch
import torch.nn as nn
from collections import OrderedDict

class VGG(nn.Module):
    """
    Standard PyTorch implementation of VGG. Pretrained imagenet model is used.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )

        # We need these for MaxUnpool operation
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()
        self.conv_ = nn.Conv2d(256, 3, 1)

    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        x = self.conv_(x)


        # x = x.view(x.size()[0], -1)
        # x = self.classifier(x)
        return x


# def get_vgg():
#     vgg = VGG()
#     temp = torchvision.models.vgg16(pretrained=True)
#     vgg.load_state_dict(temp.state_dict())
#     return vgg

if __name__ == "__main__":
    d_img = torch.rand([1, 3, 512, 512])
    model = VGG()

    result = model(d_img)
    # result = conv_enc(d_img)
    print(result.shape)