"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
import jittor.nn as nn
import jittor.nn as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def execute(self, x: jt.Var, target: jt.Var) -> jt.Var:
        if target.ndim <= 1:
            target = target.broadcast(x, [1])
            target = (target.index(1) == target).unary(op='float32')
        # logprobs = x - x.exp().sum(1).log()
        logprobs = nn.log_softmax(x, dim=-1)
        nll_loss = -(logprobs * target).sum(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
        

class CrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def execute(self, x: jt.Var, target: jt.Var) -> jt.Var:
        if target.ndim <= 1:
            target = target.broadcast(x, [1])
            target = target.index(1) == target
        x = x - x.max([1], keepdims=True)
        logsum = x.exp().sum(1).log()
        loss = logsum - (x * target).sum(1)
        return loss.mean()
        

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def execute(self, x: jt.Var, target: jt.Var) -> jt.Var:
        loss = jt.sum(-target * (x - x.exp().sum(-1).log()), dim=-1)
        return loss.mean()