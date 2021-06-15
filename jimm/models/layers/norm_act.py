"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
from jittor import nn
import jittor.nn as F

from .create_act import get_act_layer


class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True,
                 apply_act=True, act_layer=nn.ReLU, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            self.act = act_layer()
        else:
            self.act = nn.Identity()

    def execute(self, x):
        N = x.shape[0]
        C = self.num_channels
        output_shape = (N, -1)
        # TODO: 3d group norm
        if x.ndim == 4:
            output_shape = x.shape
        assert C % self.num_groups == 0
        x = x.reshape((N, self.num_groups, C // self.num_groups, -1))
        xmean = jt.mean(x, dims=[2, 3]).reshape((N, self.num_groups, 1))
        x2mean = jt.mean(x * x, dims=[2, 3]).reshape((N, self.num_groups, 1))
        xvar = (x2mean - xmean * xmean).maximum(0.0)

        if self.affine:
            w = self.weight.reshape((1, self.num_groups, -1))
            b = self.bias.reshape((1, self.num_groups, -1))
        else:
            w = 1
            b = 0
        w = w / jt.sqrt(xvar + self.eps)
        b = b - xmean * w
        x = x * w.broadcast(x, [3]) + b.broadcast(x, [3])
        x = x.reshape(output_shape)
        x = self.act(x)
        return x


# class BatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super().__init__(num_features, eps, momentum)
#
#     def execute(self, x):
#         x = x.unary('float64')
#         dims = [0] + list(range(2, x.ndim))
#         if self.is_training():
#             xmean = jt.mean(x, dims=dims)
#             x2mean = jt.mean(x * x, dims=dims)
#             if self.sync and jt.in_mpi:
#                 xmean = xmean.mpi_all_reduce("mean")
#                 x2mean = x2mean.mpi_all_reduce("mean")
#
#             xvar = (x2mean - xmean * xmean).maximum(0.0)
#             w = self.weight / jt.sqrt(xvar + self.eps)
#             b = self.bias - xmean * w
#             norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
#
#             self.running_mean.update(self.running_mean +
#                                      (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
#             self.running_var.update(self.running_var +
#                                     (xvar.reshape((-1,)) - self.running_var) * self.momentum)
#             return norm_x.unary('float32')
#         else:
#             w = self.weight / jt.sqrt(self.running_var + self.eps)
#             b = self.bias - self.running_mean * w
#             norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
#             return norm_x.unary('float32')
BatchNorm2d = nn.BatchNorm2d
