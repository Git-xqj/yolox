import torch
import torch.nn as nn
from darknet import BaseConv, DWConv, CSPLayer, CSPDarknet


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=('dark3', 'dark4', 'dark5'),
                 in_channels=None, depthwise=False, act='silu'):
        super(YOLOPAFPN, self).__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024]
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 不改变通道和大小
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)

        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

    def forward(self, input):
        out_features = self.backbone.forward(input)
        features = [out_features[f] for f in self.in_features]
        [feat1, feat2, feat3] = features

        p5 = self.lateral_conv0(feat3)
