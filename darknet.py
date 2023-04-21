import torch
from torch import nn


# 主干网络

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(name, inplace=True):
    if name == "silu":
        module = SiLU
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# 卷积 + BN + 激活
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1,
                 bias=False, act='silu'):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad,
                              groups=groups, bias=bias)
        # batch normal 在batch层面做标准化
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, kernel_size=kernel_size,
                              stride=stride, groups=in_channels, act=act)

        self.pconv = BaseConv(in_channels, out_channels, kernel_size=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


# 构造残差卷积块

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


# 提高感受野
class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 9, 13), act='silu'):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_size])
        conv2_channels = hidden_channels * (len(kernel_size) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        # x [N, C, H, W] dim=1是在C上拼接
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


#   相当于一个大的残差结构
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        # ch_in, ch_out, numbers, shortcut, expansion
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
                       for _ in range(n)]

        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)

        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


#   在一张图片中每隔一个像素拿到一个值，获得四个独立的层

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act='silu'):
        super().__init__()
        self.conv = BaseConv(in_channels, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        # 对特征层进行切片，再拼接
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)

        # 图片长宽变为原来的二分之一， 通道数翻了四倍
        return self.conv(x)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5'), depthwise=False, act='silu'):
        super(CSPDarknet, self).__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64

        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, kernel_size=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, stride=2, act=act),
            CSPLayer(base_channels * 2, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act)
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act)
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act)
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, act=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x

        x = self.dark2(x)
        outputs['dark2'] = x

        x = self.dark3(x)
        outputs['dark3'] = x

        x = self.dark4(x)
        outputs['dark4'] = x

        x = self.dark5(x)
        outputs['dark5'] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}
