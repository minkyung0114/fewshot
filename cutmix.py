import numpy as np
import torch


def rand_bbox(size,lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1.-lam) #패치크기의 비율정하기
    cut_w = np.int(W*cut_rat) #패치 너비
    cut_h = np.int(H*cut_rat) #패치 높이

    #uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx-cut_w // 2, 0, W)
    bby1 = np.clip(cy-cut_h // 2, 0, H)
    bbx2 = np.clip(cx+cut_w // 2, 0, W)
    bby2 = np.clip(cy+cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
'''
lam = np.random.beta(1.0, 1.0)
rand_index = torch.randperm()
'''