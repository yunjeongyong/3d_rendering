# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, scale, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners)
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.up = Interpolate(scale=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class UNet_(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet(nn.Module):
    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
         #   return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #else:
        # print('Unet',self.model.state_dict()['up4.conv.conv.4.bias'])

        return self.model(input)

    def __init__(self, input_channels, output_channels, gpu_ids=None):
        super(UNet, self).__init__()
        self.model = UNet_(input_channels, output_channels)
        self.gpu_ids = gpu_ids


if __name__ == '__main__':
    model = UNet_(3, 8)
    input_data = torch.ones(1, 3, 256, 256)
    output_data = model(input_data)
    print(output_data.shape)
    print('sdlfjsdfd')

#
# if __name__ == "__main__":
#     # config file
#     config = Config({
#         # device
#         "GPU_ID": "2",
#         "num_workers": 0,
#
#         # data
#         "db_path": "/media/mnt/dataset",
#         # "SMPL_path":"/media/mnt/Project/data/i_cliff_hr48.npz",
#         "SMPL_path": "/media/mnt/dataset/image_pad_cliff_padding_hr48.npz",
#         "snap_path": "/media/mnt/Project/data",  # path for saving weights
#         "save_img_path": "/media/mnt/Project/data/rgb_img",
#         "ra_body_path": "/media/mnt/Project/data/ra_body.pkl",
#         "train_size": 0.8,
#         "scenes": "all",
#
#         # ensemble in validation phase
#         "test_ensemble": True,
#         "n_ensemble": 5,
#         # learning rate 빼기, encoder만
#         # optimization
#         "batch_size": 1,
#         "learning_rate": 1e-5,
#         "weight_decay": 1e-5,
#         "n_epoch": 300,
#         "val_freq": 1,
#         "save_freq": 1,
#         "save_freq_model": 5,
#         "checkpoint": None,  # load pretrained weights
#         "T_max": 50,  # cosine learning rate period (iteration)
#         "eta_min": 0  # mininum learning rate
#     })
#     config.device = torch.device("cuda:%s" % config.GPU_ID if torch.cuda.is_available() else "cpu")
#     save_path = '/media/mnt/Project'
#     save_path_name = os.path.join(save_path, 'test.jpg')
#     img_path = '/media/mnt/Project/images.jpg'
#     image = Image.open(img_path)
#     # imgg = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     convert_tensor = transforms.ToTensor()
#     convert_size = transforms.Resize((513, 513))
#     img = convert_tensor(image).to(config.device)
#     img = convert_size(img).to(config.device)
#     img = img.unsqueeze(0)
#
#     # img = torch.rand(1, 4, 513, 513).to(config.device)
#     model = UNet(n_channels=3, n_classes=3, bilinear=False).to(config.device)
#     img_rendering = model(img).to(config.device)
#     # save_image(img_rendering, save_path_name)
#     img_rendering = img_rendering.detach()
#     img_rendering = torch.clip(img_rendering, 0., 1.)
#     img_t = img_rendering.permute(0, 2, 3, 1)
#     img_t = img_t.squeeze(0)
#     img_t = img_t.cpu().numpy()
#     plt.imshow(img_t)
#     plt.show()
#     print('img_rendering', img_rendering.shape)
#     print(0)
#
#     Image \
#         .fromarray((img_t * 255).astype(np.uint8)) \
#         .save(os.path.join(save_path, 'images_test.jpg'))