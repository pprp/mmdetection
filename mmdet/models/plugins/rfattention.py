from collections import namedtuple

import torch
import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS

from .autorf.components import SE
from .autorf.spaces import OPS

Genotype = namedtuple("Genotype", "normal normal_concat")

@PLUGIN_LAYERS.register_module() 
class ReceptiveFieldAttention(nn.Module):
    def __init__(self, in_channels, steps=3, reduction=4, se=True):
        super(ReceptiveFieldAttention, self).__init__()
        self._ops = nn.ModuleList()
        self._C = in_channels
        self._steps = steps
        self._stride = 1
        self._se = se
        self.C_in = in_channels
        self.conv3x3 = False
        self.reduction = reduction
        self.genotype = Genotype(normal=[('strippool', 0), ('avg_pool_3x3', 0), 
                                             ('avg_pool_5x5', 1), ('avg_pool_7x7', 0), ('strippool', 2), ('noise', 1)], 
                                     normal_concat=range(0, 4)) 

        op_names, indices = zip(*self.genotype.normal)
        concat = self.genotype.normal_concat

        self.bottle = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1,
                                stride=1, padding=0, bias=False)

        self.conv1x1 = nn.Conv2d(
            in_channels // self.reduction * self._steps, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        if self._se:
            self.se = SE(self.C_in, reduction=reduction)

        if self.conv3x3:
            self.conv3x3 = nn.Conv2d(
                in_channels // self.reduction * self._steps, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self._compile(in_channels, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C // self.reduction, 1, True)
            self._ops += [op]

        self.indices = indices

    def forward(self, x):
        t = self.bottle(x)

        states = [t]
        offset = 0

        total_step = (1+self._steps) * self._steps // 2

        for i in range(total_step):
            h = states[self.indices[i]]
            ops = self._ops[i]
            s = ops(h)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)

        if self.conv3x3:
            node_out = self.conv3x3(node_out)
        else:
            node_out = self.conv1x1(node_out)

        # shortcut
        node_out = node_out + x

        if self._se:
            node_out = self.se(node_out)

        return node_out
