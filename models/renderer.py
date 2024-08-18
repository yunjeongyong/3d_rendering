
import torch.nn as nn
import torch
import cv2
from torchvision.utils import save_image


class VGG_renderer(nn.Module):
    def __init__(self, num_clsses: int = 1000, init_weights: bool = True):
        super(VGG_renderer, self).__init__()
        self.convnet = nn.Sequential(
            # Input Channel (RGB: 3)
            nn.ConvTranspose2d(in_channels=3, out_channels=256, kernel_size=1),
            # nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True))

    def forward(self, x:torch.Tensor):
        x = self.convnet(x)
        # x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    raster_img_path = '/media/mnt/Project/fadsfsdf.jpg'
    save_raster_img = '/media/mnt/Project/fadsfsdf_save.jpg'
    d_img = torch.rand(1, 3, 64, 64)
    raster_img_path = cv2.imread(raster_img_path)
    raster_img_path = torch.Tensor(raster_img_path)
    raster_img_path = raster_img_path.permute(2, 0, 1)
    raster_img_path = raster_img_path.unsqueeze(0)
    # conv_enc = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
    # d_img = conv_enc(d_img)
    # t_img = torch.rand(1, 32, 128, 128)
    model = VGG_renderer()
    # model = TransformerDecoderUnit(feat_dim=32, n_head=8, pos_en_flag=True, attn_type='softmax')
    result = model(raster_img_path)
    save_image(result, save_raster_img)
    # result = conv_enc(d_img)
    print(result.shape)
    # model = VGG_s()
    # d_img = model(d_img)
    # print(d_img.shape)
