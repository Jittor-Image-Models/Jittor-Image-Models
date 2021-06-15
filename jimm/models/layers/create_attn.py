"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
from jittor import nn


def get_attn(attn_type):
    if isinstance(attn_type, nn.Module):
        return attn_type
    module_cls = None
    return module_cls


def create_attn(attn_type, channels, **kwargs):
    module_cls = get_attn(attn_type)
    if module_cls is not None:
        # NOTE: it's expected the first (positional) argument of all attention layers is the # input channels
        return module_cls(channels, **kwargs)
    return None
