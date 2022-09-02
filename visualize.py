import os.path

import torchvision.utils

from fewshot import SiameseNetwork, ContrastiveLoss, build_dataloader,Face_Dataset
from common import *
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

transformers = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))
                                   ])


train_dataset = Face_Dataset(data_dir,phase="train",transformer=transformers)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)


def train_data_show(train_loader):

    for i in range(1,5):

        faceA, faceB, other = train_loader[i]
        other2name = {0: "same_person", 1: "other_person"}
        if other == 0:
            other = other2name[0]
        else:
            other = other2name[1]

        #print(f"faceA:{faceA.shape}")
        faceA = faceA.numpy().transpose(1, 2, 0)
        #print(f"faceA:{faceA.shape}")
        faceB = faceB.numpy().transpose(1, 2, 0)
        plt.figure(figsize=(6, 6))
        plt.subplot(121)
        plt.title("faceA")
        plt.imshow(faceA, cmap='gray')
        plt.subplot(122)
        plt.title(f"faceB_{other}")
        plt.imshow(faceB, cmap='gray')
        plt.savefig(os.path.join(SAVE_DIR,f"train_sample_img{i}.png"),dpi=100)
        plt.show()




#train_data_show(train_dataset)


def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    #cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutMix_traindata_show(train_loader):
    for batch_idx, (faceA, faceB, other) in enumerate(train_loader):


        faceA = faceA
        faceB = faceB
        other = other
        r = np.random.rand(1)
        beta = 1.
        if beta > int(0.0) and r < .5:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(faceB.size()[0])

            target_a = other
            target_b = other[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(faceB.size(), lam)
            '''
            print(f"bbx1:{bbx1}")
            print(f"bby1:{bby1}")
            print(f"bbx2:{bbx2}")
            print(f"bby2:{bby2}")
            '''
            faceB[:, :, bbx1:bbx2, bby1:bby2] = faceB[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (faceB.size()[-1] * faceB.size()[-2]))


            plt.title("cutMix data images")
            plt.imshow(torchvision.utils.make_grid(faceB, normalize=True).permute(1,2,0))
            plt.savefig(os.path.join(SAVE_DIR, f"cutmix_augmentation.png"), dpi=100)
            plt.show()





#cutMix_traindata_show(train_loader)




