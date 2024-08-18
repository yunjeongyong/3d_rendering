import torch.nn as nn
import torch
import torch.nn.functional as F

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
            nn.Conv2d(ngf, 3, 2, 2, 0),
            nn.Conv2d(3, 3, 2, 2, 0)
        )

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

class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim=None, mode: str=''):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.op = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)
        if mode == 'ps':
            self.op = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        return self.op(x)

class DwSample(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.op = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.op(x)

if __name__ == '__main__':
    model = Decoder(8, 64)
    input_data = torch.ones(2, 8, 128, 128)
    output_data = model(input_data)
    print(output_data.shape)
