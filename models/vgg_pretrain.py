# # import torch
# # import torch.nn as nn
# #
# #
# # class VGG_A(nn.Module):
# #     def __init__(self, num_classes: int = 1000, init_weights: bool = True):
# #         super(VGG_A, self).__init__()
# #         self.convnet = nn.Sequential(
# #             # Input Channel (RGB: 3)
# #             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
# #
# #             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
# #
# #             # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
# #             # nn.ReLU(inplace=True),
# #             # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
# #             # nn.ReLU(inplace=True),
# #             # nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
# #
# #             # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
# #             # nn.ReLU(inplace=True),
# #             # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
# #             # nn.ReLU(inplace=True),
# #             # nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
# #
# #             # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
# #             # nn.ReLU(inplace=True),
# #             # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
# #             # nn.ReLU(inplace=True),
# #             # nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
# #         )
# #
# #         self.fclayer = nn.Sequential(
# #             nn.Linear(512 * 7 * 7, 4096),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(p=0.5),
# #             nn.Linear(4096, 4096),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(p=0.5),
# #             nn.Linear(4096, num_classes),
# #             # nn.Softmax(dim=1), # Loss인 Cross Entropy Loss 에서 softmax를 포함한다.
# #         )
# #
# #     def forward(self, x: torch.Tensor):
# #         x = self.convnet(x)
# #         # x = torch.flatten(x, 1)
# #         # x = self.fclayer(x)
# #         return x
# #
# #
# # if __name__ == "__main__":
# #     d_img = torch.rand([1, 3, 512, 512])
# #     model = VGG_A()
# #     print(model)
# #
# #     result = model(d_img)
# #     # result = conv_enc(d_img)
# #     print(result.shape)
#
#
# import torch.nn as nn
# import torch
# from model.net_utils import PosEnSine, softmax_attention, dotproduct_attention, long_range_attention, \
#     short_range_attention, patch_attention
#
#
# class VGG_s(nn.Module):
#     def __init__(self, num_clsses: int = 1000, init_weights: bool = True):
#         super(VGG_s, self).__init__()
#         self.convnet = nn.Sequential(
#             # Input Channel (RGB: 3)
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
#
#             # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
#             # nn.ReLU(inplace=True),
#         )
#             # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
#             # nn.ReLU(inplace=True),)
#             # nn.MaxPool2d(kernel_size=2, stride=2)) # 56 -> 28
#
#     def forward(self, x:torch.Tensor):
#         x = self.convnet(x)
#         # x = torch.flatten(x, 1)
#         return x
#
#
#
# class OurMultiheadAttention(nn.Module):
#     def __init__(self, feat_dim, n_head, d_k=None, d_v=None):
#         super(OurMultiheadAttention, self).__init__()
#         if d_k is None:
#             d_k = feat_dim // n_head
#         if d_v is None:
#             d_v = feat_dim // n_head
#
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#
#         # pre-attention projection
#         self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
#         self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
#         self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)
#
#         # after-attention combine heads
#         self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)
#
#     def forward(self, q, k, v, attn_type='softmax', **kwargs):
#         # input: b x d x h x w
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#
#         # Pass through the pre-attention projection: b x (nhead*dk) x h x w
#         # Separate different heads: b x nhead x dk x h x w
#         q = self.w_qs(q).view(q.shape[0], n_head, d_k, q.shape[2], q.shape[3])
#         k = self.w_ks(k).view(k.shape[0], n_head, d_k, k.shape[2], k.shape[3])
#         v = self.w_vs(v).view(v.shape[0], n_head, d_v, v.shape[2], v.shape[3])
#
#         # -------------- Attention -----------------
#         if attn_type == 'softmax':
#             q, attn = softmax_attention(q, k, v)  # b x n x dk x h x w --> b x n x dv x h x w
#         elif attn_type == 'dotproduct':
#             q, attn = dotproduct_attention(q, k, v)
#         elif attn_type == 'patch':
#             q, attn = patch_attention(q, k, v, P=kwargs['P'])
#         elif attn_type == 'sparse_long':
#             q, attn = long_range_attention(q, k, v, P_h=kwargs['ah'], P_w=kwargs['aw'])
#         elif attn_type == 'sparse_short':
#             q, attn = short_range_attention(q, k, v, Q_h=kwargs['ah'], Q_w=kwargs['aw'])
#         else:
#             raise NotImplementedError(f'Unknown attention type {attn_type}')
#         # ------------ end Attention ---------------
#
#         # Concatenate all the heads together: b x (n*dv) x h x w
#         q = q.reshape(q.shape[0], -1, q.shape[3], q.shape[4])
#         q = self.fc(q)  # b x d x h x w
#
#         return q, attn
#
#
# class TransformerEncoderUnit(nn.Module):
#     def __init__(self, feat_dim, n_head=4, pos_en_flag=True, attn_type='softmax', P=None):
#         super(TransformerEncoderUnit, self).__init__()
#
#         self.vgg_s = VGG_s()
#         self.feat_dim = feat_dim
#         self.attn_type = attn_type
#         self.pos_en_flag = pos_en_flag
#         self.P = P
#
#         self.pos_en = PosEnSine(self.feat_dim // 2)
#         self.attn = OurMultiheadAttention(feat_dim, n_head)
#
#         self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
#         self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
#         self.activation = nn.ReLU(inplace=True)
#
#         self.norm1 = nn.BatchNorm2d(self.feat_dim)
#         self.norm2 = nn.BatchNorm2d(self.feat_dim)
#         self._conv = nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1)
#
#     def forward(self, src):
#         src = self.vgg_s(src)
#         if self.pos_en_flag:
#             pos_embed = self.pos_en(src)
#         else:
#             pos_embed = 0
#
#         # multi-head attention
#         src2 = self.attn(q=src + pos_embed, k=src + pos_embed, v=src, attn_type=self.attn_type, P=self.P)[0]
#         src = src + src2
#         src = self.norm1(src)
#
#         # feed forward
#         src2 = self.linear2(self.activation(self.linear1(src)))
#         src = src + src2
#         src = self.norm2(src)
#         src = self._conv(src)
#
#         return src
#
#
#
# if __name__ == "__main__":
#     d_img = torch.rand(1, 3, 512, 512)
#     # conv_enc = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
#     # d_img = conv_enc(d_img)
#     # t_img = torch.rand(1, 32, 128, 128)
#     model = TransformerEncoderUnit(feat_dim=128, n_head=4, pos_en_flag=True, attn_type='softmax')
#     # model = TransformerDecoderUnit(feat_dim=32, n_head=8, pos_en_flag=True, attn_type='softmax')
#     result = model(d_img)
#     # result = conv_enc(d_img)
#     print(result.shape)
#     # model = VGG_s()
#     # d_img = model(d_img)
#     # print(d_img.shape)
#
from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch.legacy import nn as nnl
import torch.utils.model_zoo as model_zoo

__all__ = ['vggm']

pretrained_settings = {
    'vggm': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth',
            'input_space': 'BGR',
            'input_size': [3, 221, 221],
            'input_range': [0, 255],
            'mean': [123.68, 116.779, 103.939],
            'std': [1, 1, 1],
            'num_classes': 1000
        }
    }
}

class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(2, 2)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((1, 1),(2, 2),(0, 0),ceil_mode=True),
            # nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
            # nn.ReLU(),
            # SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            # nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            # nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            # nn.ReLU(),
            # nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        )
        self.classif = nn.Sequential(
            nn.Linear(18432,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classif(x)
        return x

def vggm(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['vggm'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = VGGM(num_classes=1000)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = VGGM(num_classes=num_classes)
    return model


if __name__ == "__main__":
    d_img = torch.rand([1, 3, 512, 512])
    model = VGGM(num_classes=1000)

    result = model(d_img)
    # result = conv_enc(d_img)
    print(result.shape)