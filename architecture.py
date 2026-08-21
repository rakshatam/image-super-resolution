import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c, 1, 1))
        self.b = nn.Parameter(torch.zeros(c, 1, 1))
        self.eps = eps
    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = (x - m).pow(2).mean(1, keepdim=True)
        return self.w * (x - m) / torch.sqrt(v + self.eps) + self.b

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw, bias=True)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw//2, dw//2, 1, 1, 0, bias=True))
        self.conv3 = nn.Conv2d(dw//2, c, 1, 1, 0, bias=True)
        ffn = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1, 1, 0, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn//2, c, 1, 1, 0, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma

class NAFNetSR(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=64, num_blks=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(in_ch, width, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blks)])
        self.up = nn.Sequential(nn.Conv2d(width, out_ch * upscale**2, 3, 1, 1, bias=True), nn.PixelShuffle(upscale))

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        fea = self.intro(x)
        fea = self.body(fea) + fea
        return self.up(fea) + base
