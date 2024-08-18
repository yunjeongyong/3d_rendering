# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import os.path
import matplotlib.pyplot as plt
from option.config import Config
from PIL import Image
from common import constants
from torchvision.utils import save_image
from typing import Any
import cv2
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "SRResNet", "Discriminator",
    "srresnet_x4", "discriminator", "content_loss",
]


class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            num_rcb: int,
            upscale_factor: int
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input image size must equal 96
        assert x.shape[2] == 96 and x.shape[3] == 96, "Image shape must equal 96x96"

        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out


class _ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            feature_model_extractor_node: str,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(_ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data. 
        # This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        loss = F_torch.mse_loss(sr_feature, gt_feature)

        return loss


def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale_factor=1, **kwargs)

    return model


def discriminator() -> Discriminator:
    model = Discriminator()

    return model


def content_loss(**kwargs: Any) -> _ContentLoss:
    content_loss = _ContentLoss(**kwargs)

    return content_loss


if __name__ == "__main__":
    # config file
    config = Config({
        # device
        "GPU_ID": "2",
        "num_workers": 0,

        # data
        "db_path": "/media/mnt/dataset",
        # "SMPL_path":"/media/mnt/Project/data/i_cliff_hr48.npz",
        "SMPL_path": "/media/mnt/dataset/image_pad_cliff_padding_hr48.npz",
        "snap_path": "/media/mnt/Project/data",  # path for saving weights
        "save_img_path": "/media/mnt/Project/data/rgb_img",
        "ra_body_path": "/media/mnt/Project/data/ra_body.pkl",
        "train_size": 0.8,
        "scenes": "all",

        # ensemble in validation phase
        "test_ensemble": True,
        "n_ensemble": 5,
        # learning rate 빼기, encoder만
        # optimization
        "batch_size": 1,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 300,
        "val_freq": 1,
        "save_freq": 1,
        "save_freq_model": 5,
        "checkpoint": None,  # load pretrained weights
        "T_max": 50,  # cosine learning rate period (iteration)
        "eta_min": 0  # mininum learning rate
    })
    config.device = torch.device("cuda:%s" % config.GPU_ID if torch.cuda.is_available() else "cpu")
    save_path = '/media/mnt/Project'
    save_path_name = os.path.join(save_path, 'test.jpg')
    # img_path = '/media/mnt/Project/rgb_img_epoch35.jpg'
    img_path = '/media/mnt/Project/images.jpg'
    image = Image.open(img_path)
    # imgg = cv2.imread(img_path, cv2.IMREAD_COLOR)
    convert_tensor = transforms.ToTensor()
    convert_size = transforms.Resize((513, 513))
    img = convert_tensor(image).to(config.device)
    img = convert_size(img).to(config.device)
    img = img.unsqueeze(0)

    # img = torch.rand(1, 4, 513, 513).to(config.device)
    model = srresnet_x4(in_channels = 3,out_channels = 3,channels = 3, num_rcb = 16).to(config.device)
    img_rendering = model(img).to(config.device)
    # save_image(img_rendering, save_path_name)
    img_t = img_rendering
    img_t = img_t.permute(0, 2, 3, 1)
    img_t = img_t.squeeze(0)
    img_t = img_t.detach().cpu().numpy()
    # im = Image.fromarray(img_t)
    # im.save('test.png')
    # cv2.imwrite('./test.png', img_t)
    plt.imshow(img_t)
    plt.show()

    print('img_rendering', img_rendering.shape)
    print(0)