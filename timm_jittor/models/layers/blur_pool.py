"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
import jittor.nn as F
from jittor import nn, misc
import numpy as np
from typing import Dict
from .padding import get_padding


class BlurPool2d(nn.Module):
    filt: Dict[str, jt.Var]

    def __init__(self, channels, filt_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        pad_size = [get_padding(filt_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)
        self._coeffs = jt.array((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs)  # for torchscript compat
        self.filt = {}  # lazy init by device for DataParallel compat

    def _create_filter(self, like: jt.Var):
        blur_filter = jt.type_as(self._coeffs[:, None] * self._coeffs[None, :], like)
        return misc.repeat(blur_filter[None, None, :, :], (self.channels, 1, 1, 1))

    def execute(self, input_tensor: jt.Var) -> jt.Var:
        C = input_tensor.shape[1]
        blur_filt = self.filt.get(str(input_tensor.device), self._create_filter(input_tensor))
        return F.conv2d(
            self.padding(input_tensor), blur_filt, stride=self.stride, groups=C)
