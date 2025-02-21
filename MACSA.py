import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TAAttention(nn.Module):
    def __init__(self, channel=512, no_spatial=False, reduction=16, L=32):
        super(TAAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
        # Initialize self.fs
        self.d = max(L, channel // reduction)
        self.fs = nn.Sequential(
            nn.Linear(channel, self.d),
            nn.ReLU(inplace=True),
            nn.Linear(self.d, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        # Apply self.fs to x_out
        b, c, h, w = x_out.size()
        x_out_flat = x_out.view(b, c, -1).mean(dim=2)
        fs_out = self.fs(x_out_flat).view(b, c, 1, 1)
        x_out = x_out * fs_out
        return x_out

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    ta = TAAttention()
    output=ta(input)
    print(output.shape)