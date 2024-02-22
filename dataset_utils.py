import torch
import numpy as np
import torch.nn.functional as F
import cv2
import random
import numpy as np
import os
import pywt
from PIL import Image

def CutFreq_StageI(im1, im2, xfm, ifm, component):
    im1_l, im1_h = xfm(im1)
    im2_l, im2_h = xfm(im2)
    im2_h[2] = im1_h[2]
    im2 = ifm((im2_l, im2_h))
    return im1, im2

def CutFreq_StageII(im1, im2, xfm, ifm, component):
    im2_l, im2_h = xfm(im2)
    
    idx = np.random.rand(1)
    zero_band = torch.zeros_like(im2_h[0])
    zero_band1 = torch.zeros_like(im2_h[1])
    zero_band2 = torch.zeros_like(im2_h[2])
    if idx <= 0.33:
        im2_h[0] = zero_band
    elif idx > 0.33 and idx <=0.66:
        im2_h[1] = zero_band1
    elif idx > 0.66:
        im2_h[2] = zero_band2
    else:
        pass
    im2 = ifm((im2_l, im2_h))
    return im1, im2

def apply_augment(im1, im2, xfm, ifm, augs, mix_p, component):
    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
    mask = None

    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "CutFreq_StageI":
        im1_aug, im2_aug = CutFreq_StageI(
            im1.clone(), im2.clone(), xfm, ifm, component)
    elif aug == "CutFreq_StageII":
        im1_aug, im2_aug = CutFreq_StageII(
            im1.clone(), im2.clone(), xfm, ifm, component)
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, mask, aug
    