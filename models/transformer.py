import torch.nn as nn
import torch
from models.net_utils import PosEnSine, softmax_attention, dotproduct_attention, long_range_attention, \
    short_range_attention, patch_attention



class VGG_s(nn.Module):
    def __init__(self, num_clsses: int = 1000, init_weights: bool = True):
        super(VGG_s, self).__init__()
        self.convnet = nn.Sequential(
            # Input Channel (RGB: 3)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 112 -> 56

            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)) # 56 -> 28

    def forward(self, x:torch.Tensor):
        x = self.convnet(x)
        print('x.shape',x.shape)
        # x = torch.flatten(x, 1)
        return x



class OurMultiheadAttention(nn.Module):
    def __init__(self, feat_dim, n_head, d_k=None, d_v=None):
        super(OurMultiheadAttention, self).__init__()
        if d_k is None:
            d_k = feat_dim // n_head
        if d_v is None:
            d_v = feat_dim // n_head

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)

    def forward(self, q, k, v, attn_type='softmax', **kwargs):
        # input: b x d x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection: b x (nhead*dk) x h x w
        # Separate different heads: b x nhead x dk x h x w
        q = self.w_qs(q).view(q.shape[0], n_head, d_k, q.shape[2], q.shape[3])
        k = self.w_ks(k).view(k.shape[0], n_head, d_k, k.shape[2], k.shape[3])
        v = self.w_vs(v).view(v.shape[0], n_head, d_v, v.shape[2], v.shape[3])

        # -------------- Attention -----------------
        if attn_type == 'softmax':
            q, attn = softmax_attention(q, k, v)  # b x n x dk x h x w --> b x n x dv x h x w
        elif attn_type == 'dotproduct':
            q, attn = dotproduct_attention(q, k, v)
        elif attn_type == 'patch':
            q, attn = patch_attention(q, k, v, P=kwargs['P'])
        elif attn_type == 'sparse_long':
            q, attn = long_range_attention(q, k, v, P_h=kwargs['ah'], P_w=kwargs['aw'])
        elif attn_type == 'sparse_short':
            q, attn = short_range_attention(q, k, v, Q_h=kwargs['ah'], Q_w=kwargs['aw'])
        else:
            raise NotImplementedError(f'Unknown attention type {attn_type}')
        # ------------ end Attention ---------------

        # Concatenate all the heads together: b x (n*dv) x h x w
        q = q.reshape(q.shape[0], -1, q.shape[3], q.shape[4])
        q = self.fc(q)  # b x d x h x w

        return q, attn


class TransformerEncoderUnit(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True, attn_type='softmax', P=None):
        super(TransformerEncoderUnit, self).__init__()

        self.vgg_s = VGG_s()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.pos_en_flag = pos_en_flag
        self.P = P

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = OurMultiheadAttention(feat_dim, n_head)

        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(self.feat_dim)
        self.norm2 = nn.BatchNorm2d(self.feat_dim)
        self._conv = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1)

    def forward(self, src):
        src = self.vgg_s(src)
        if self.pos_en_flag:
            pos_embed = self.pos_en(src)
        else:
            pos_embed = 0

        # multi-head attention
        src2 = self.attn(q=src + pos_embed, k=src + pos_embed, v=src, attn_type=self.attn_type, P=self.P)[0]
        src = src + src2
        src = self.norm1(src)

        # feed forward
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        src = self._conv(src)
        # src = nn.functional.interpolate(src, (128, 128), mode='bilinear')

        return src



if __name__ == "__main__":
    d_img = torch.rand(1, 3, 512, 512)
    # conv_enc = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
    # d_img = conv_enc(d_img)
    # t_img = torch.rand(1, 32, 128, 128)
    model = TransformerEncoderUnit(feat_dim=64, n_head=4, pos_en_flag=True, attn_type='softmax')
    # model = TransformerDecoderUnit(feat_dim=32, n_head=8, pos_en_flag=True, attn_type='softmax')
    result = model(d_img)
    # result = conv_enc(d_img)
    print(result.shape)
    # model = VGG_s()
    # d_img = model(d_img)
    # print(d_img.shape)

