import torch
from torch import nn

from module.BaseBlocks import BasicConv2d

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):

        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))
        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)


    def forward(self, in_feat):

        down_feats = self.down(in_feat)
        out_feats = []

        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class DenseTransLayer(nn.Module):
    def __init__(self, in_C, out_C):
        super(DenseTransLayer, self).__init__()
        down_factor = in_C // out_C
        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb):
        feat = self.fuse_down_mul(rgb)
        return self.fuse_main(self.res_main(feat) + feat)

class transDDPM(nn.Module):
    def __init__(self, in_xC,out_C):
        super().__init__()
        self.down_input = nn.Conv2d(out_C, out_C//4, 1)
        self.unfold1 = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.unfold3 = nn.Unfold(kernel_size=3, dilation=3, padding=3, stride=1)
        self.unfold5 = nn.Unfold(kernel_size=3, dilation=5, padding=5, stride=1)
        self.fuse = BasicConv2d(out_C, out_C, 3, 1, 1)


    def forward(self, x, y):
        x = self.down_input(x)
        N, xC, xH, xW = x.size()
        unfold_x1 = self.unfold1(x).reshape([N, xC, -1, xH, xW])
        unfold_x2 = self.unfold3(x).reshape([N, xC, -1, xH, xW])
        unfold_x3 = self.unfold5(x).reshape([N, xC, -1, xH, xW])

        y0 = y[0].reshape([N, xC, 9, xH, xW])
        y1 = y[1].reshape([N, xC, 9, xH, xW])
        y2 = y[2].reshape([N, xC, 9, xH, xW])
        result1 = (unfold_x1 * y0).sum(2)
        result2 = (unfold_x2 * y1).sum(2)
        result3 = (unfold_x3 * y2).sum(2)
        return self.fuse(torch.cat((x, result1, result2, result3), dim=1))

