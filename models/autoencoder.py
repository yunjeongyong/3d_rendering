

import torch.nn as nn
import torch
from option.config import Config
import torch.nn.functional as F


device = torch.device("cuda:%s" %"0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % "0")
else:
    print('Using CPU')

class Encoder(nn.Module):
    def __init__(self, ngf, d_model):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(4, ngf, 1, 1, 0),
            ResnetBlock(ngf * 1, ngf * 1, 3, 1, 1, num_res=1), DwSample(ngf * 1),
            ResnetBlock(ngf * 1, ngf * 2, 3, 1, 1, num_res=1), DwSample(ngf * 2),
            ResnetBlock(ngf * 2, ngf * 4, 3, 1, 1, num_res=1), DwSample(ngf * 4),
            ResnetBlock(ngf * 4, ngf * 4, 3, 1, 1, num_res=1),
            nn.GroupNorm(num_groups=32, num_channels=ngf * 4, eps=1e-6, affine=True),
            nn.Conv2d(ngf * 4, d_model, 3, 1, 1)
        )

    def forward(self, x):
        return self.op(x)


class Decoder(nn.Module):
    def __init__(self, d_model, ngf):
        super(Decoder, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(d_model, ngf * 4, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=64 * 4, eps=1e-6, affine=True),
            ResnetBlock(ngf * 4, ngf * 4, 1, 1, 0, num_res=1), UpSample(ngf * 4),
            ResnetBlock(ngf * 4, ngf * 2, 1, 1, 0, num_res=1), UpSample(ngf * 2),
            ResnetBlock(ngf * 2, ngf * 1, 1, 1, 0, num_res=1), UpSample(ngf * 1),
            ResnetBlock(ngf * 1, ngf * 1, 1, 1, 0, num_res=1),
            nn.Conv2d(ngf, 3, 1, 1, 0)
        )

    def forward(self, x):
        return self.op(x)

class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim=None, mode: str=''):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.op = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)
        if mode == 'ps':
            self.op = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        return self.op(x)



class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, num_res):
        super().__init__()
        self.num_res = num_res
        for idx in range(self.num_res):
            self.add_module("norm_head_{}".format(idx),
                            nn.GroupNorm(num_groups=32, num_channels=in_dim, eps=1e-6, affine=True))
            self.add_module("norm_tail_{}".format(idx),
                            nn.GroupNorm(num_groups=32, num_channels=out_dim, eps=1e-6, affine=True))
            self.add_module("op_head_{}".format(idx), nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding))
            self.add_module("op_tail_{}".format(idx), nn.Conv2d(out_dim, out_dim, kernel_size, stride, padding))
            self.add_module("short_cut_{}".format(idx), nn.Conv2d(in_dim, out_dim, 3, 1, 1))
            in_dim = out_dim

    def forward(self, x):
        for idx in range(self.num_res):
            h = x
            h = getattr(self, "norm_head_{}".format(idx))(h)
            h = h * torch.sigmoid(h)  # swish
            h = getattr(self, "op_head_{}".format(idx))(h)
            h = getattr(self, "norm_tail_{}".format(idx))(h)
            h = h * torch.sigmoid(h)
            h = getattr(self, "op_tail_{}".format(idx))(h)
            x = h + getattr(self, "short_cut_{}".format(idx))(x)
        return x

class DwSample(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.op = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.op(x)

class YoonNet(nn.Module):
    def __init__(self, config, ngf, dim, a):
        super(YoonNet, self).__init__()
        self.recon_enc = Encoder(ngf, dim)
        self.recon_dec = Decoder(dim, ngf)

    def forward(self, img):
        render_img = self.recon_enc(img)
        render_img = self.recon_dec(render_img)

        return render_img

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

    img = torch.rand(1, 4, 513, 513).to(config.device)
    model = YoonNet(config=config, ngf=64, dim=256, a=4).to(config.device)
    img_rendering = model(img)
    print('img_rendering', img_rendering.shape)
    print(0)