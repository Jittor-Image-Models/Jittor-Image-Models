"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import random
import math
import numpy as np


class RandomMixup:

    def __init__(self, num_classes, p=0.5, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        
    def __call__(self, batch, target):
        if target.ndim == 1:
            target = np.eye(self.num_classes, dtype=np.float32)[target]
            
        if random.random() >= self.p:
            return batch, target
            
        batch_rolled = np.roll(batch, 1, 0)
        target_rolled = np.roll(target, 1, 0)
        
        lambda_param = np.random.beta(self.alpha, self.alpha)
        batch_rolled *= 1 - lambda_param
        batch = batch * lambda_param + batch_rolled
        
        target_rolled *= (1 - lambda_param)
        target = target * lambda_param + target_rolled
        
        return batch, target
        

class RandomCutmix:

    def __init__(self, num_classes, p=0.5, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        
    def __call__(self, batch, target):
        if target.ndim == 1:
            target = np.eye(self.num_classes, dtype=np.float32)[target]
            
        if random.random() >= self.p:
            return batch, target
            
        batch_rolled = np.roll(batch, 1, 0)
        target_rolled = np.roll(target, 1,0)
        
        lambda_param = np.random.beta(self.alpha, self.alpha)
        W, H = batch.shape[-1], batch.shape[-2]
        
        r_x = np.random.randint(W)
        r_y = np.random.randint(H)
        
        r = 0.5 * math.sqrt(1. - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)
        
        x1 = int(max(r_x - r_w_half, 0))
        y1 = int(max(r_y - r_h_half, 0))
        x2 = int(min(r_x + r_w_half, W))
        y2 = int(min(r_y + r_h_half, H))
        
        batch[:,:,y1:y2,x1:x2] = batch_rolled[:,:,y1:y2,x1:x2]
        lambda_param = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        
        target_rolled *= 1 - lambda_param
        target = target * lambda_param + target_rolled
        
        return batch, target


class RandomChoice:

    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p
    
    def __call__(self, *args):
        t = random.choices(self.transforms, weights=self.p)[0]
        return t(*args)