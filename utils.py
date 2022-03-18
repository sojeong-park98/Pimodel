import math
import torch

class AddGaussianNoise(object):
    def __init__(self, p, mean=0., std=1.):
        self.p = p
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            # 랜덤 가우시안 노이즈 생성
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor