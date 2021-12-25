'''
@Author: Jiangtao
@Date: 2020-03-02 16:27:26
@LastEditors: Jiangtao
@LastEditTime: 2020-03-02 17:14:43
@Description: 
'''
import torch.nn.functional as F
import os
import torch
import sys
from fmix import *

class FMix(FMixBase):
    r""" FMix augmentation

        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].

        Example
        -------

        .. code-block:: python

            class FMixExp(pl.LightningModule):
                def __init__(*args, **kwargs):
                    self.fmix = Fmix(...)
                    # ...

                def training_step(self, batch, batch_idx):
                    x, y = batch
                    x = self.fmix(x)

                    feature_maps = self.forward(x)
                    logits = self.classifier(feature_maps)
                    loss = self.fmix.loss(logits, y)

                    # ...
                    return loss
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def __call__(self, x):
        # Sample mask and generate random permutation
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        # Mix the images
        x1 = mask * x
        x2 = (1 - mask) * x[index]
        self.index = index
        self.lam = lam
        return x1+x2

    def loss(self, y_pred, y, train=True):
        return fmix_loss(y_pred, y, self.index, self.lam, train, self.reformulate)

if __name__ == '__main__':

    lam, mask = sample_mask(alpha=1, decay_power=3,shape=(128,128))
    print(lam)
    print(mask)
    print(type(mask))
    print(mask.shape)