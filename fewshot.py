import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from common import *

phase = "train"

'''
person_items = []

for (root, dirs, files) in os.walk(os.path.join(data_dir,phase),topdown=True):

    if len(files) > 0:
        for file_name in files:
            person_items.append((os.path.join(root, file_name)))

print(f" len person items : {len(person_items)}")
print(f" person items : {person_items}")

index = 0
faceA_path = person_items[index]

print(Path(faceA_path).parent)
print(Path(faceA_path).parent.name)

person = Path(faceA_path).parent.name
print(f"person :{person}")
same_person_dir = Path(faceA_path).parent
print(f"same person :{same_person_dir}")


same_person_images = [img for img in os.listdir(same_person_dir) if img.endswith('.png')]
print(f"same_person_images:{same_person_images}")

np.random.choice(same_person_images)

faceB_path = os.path.join(same_person_dir, np.random.choice(same_person_images))

faceA_image = cv2.imread(faceA_path)
faceB_image = cv2.imread(faceB_path)

plt.figure()
plt.subplot(121)
plt.title("sample")
plt.imshow(faceA_image, cmap='gray')
plt.subplot(122)
plt.title("sample-positive")
plt.imshow(faceB_image, cmap='gray')
plt.savefig(os.path.join(data_dir,'samplePositive.png'), dpi = 100)
#plt.show()

while True:
    faceB_path = np.random.choice(person_items)
    if person != Path(faceB_path).parent.name:
        break

faceA_image = cv2.imread(faceA_path)
faceB_image = cv2.imread(faceB_path)

plt.figure()
plt.subplot(121)
plt.title("sample")
plt.imshow(faceA_image, cmap='gray')
plt.subplot(122)
plt.title("sample-negative")
plt.imshow(faceB_image, cmap='gray')
plt.savefig(os.path.join(data_dir,'sampleNegative.png'), dpi=100)
#plt.show()
'''

class Face_Dataset():
    def __init__(self, data_dir, phase, transformer=None):
        self.person_items = []
        for (root, dirs, files) in os.walk(os.path.join(data_dir, phase)):
            if len(files) > 0:
                for file_name in files:
                    self.person_items.append(os.path.join(root, file_name))

        self.transformer = transformer

    def __len__(self):
        return len(self.person_items)

    def __getitem__(self, index ):
        faceA_path = self.person_items[index]
        person = Path(faceA_path).parent.name
        same_person = np.random.randint(2)

        if same_person:
            same_person_dir = Path(faceA_path).parent
            same_person_fn = [fn for fn in os.listdir(same_person_dir) if fn.endswith("png")]
            faceB_path = os.path.join(same_person_dir, np.random.choice(same_person_fn))
        else:
            while True:
                faceB_path = np.random.choice(self.person_items)
                if person != Path(faceB_path).parent.name:
                    break

        faceA_image = cv2.imread(faceA_path, 0)
        faceB_image = cv2.imread(faceB_path, 0)

        if self.transformer:
            faceA_image = self.transformer(faceA_image)
            faceB_image = self.transformer(faceB_image)

        return faceA_image, faceB_image, np.array([1 - same_person])


'''
train_dataset = Face_Dataset(data_dir,phase="train")

faceA, faceB, other = train_dataset[0]
plt.figure(figsize=(6, 6))
plt.subplot(121)
plt.imshow(faceA, cmap='gray')
plt.subplot(122)
plt.imshow(faceB, cmap='gray')
plt.show()
print(other)
'''

def build_transformer(image_size =100):
    transformers = {}
    transformers["train"] = transforms.Compose([transforms.ToPILImage(),
                                                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.Resize((image_size, image_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5), (0.5))
                                                ])
    transformers["val"] = transforms.Compose([ transforms.ToPILImage(),
                                               transforms.Resize((image_size, image_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5), (0.5))
                                               ])

    return transformers


def build_dataloader(data_dir, batch_size=64):

    dataloaders = {}

    transformers = build_transformer()
    tr_dataset = Face_Dataset(data_dir, phase="train", transformer=transformers["train"])
    dataloaders["train"] = DataLoader(tr_dataset, shuffle=True, batch_size=batch_size)

    val_dataset = Face_Dataset(data_dir, phase="val", transformer=transformers["val"])
    dataloaders["val"] = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    return dataloaders



'''
for _ in range(3):
    for phase in ["train", "val"]:
        for index, batch in enumerate(dataloaders[phase]):
            faceAs = batch[0]
            faceBs = batch[1]
            others = batch[2]

            if index % 100 == 0:
                print(f"{phase} - {index}/{len(dataloaders[phase])}")
                
                
'''

def convBlock(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channel),
    )

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential( convBlock(1,4),
                                       convBlock(4,8),
                                       convBlock(8,8),
                                       nn.Flatten(),
                                       nn.Linear(8*100*100, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, 256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, 10)
                                       )

    def forward(self, x1, x2):
        out1 = self.features(x1)
        out2 = self.features(x2)
        return out1, out2


x1 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
x2 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)

model = SiameseNetwork()

out1, out2 = model(x1, x2)

print(f'out1:{out1.shape}')
print(f'out2:{out2.shape}')

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):

        super().__init__()
        self.margin = margin


    def forward(self, z1, z2, label):

        dist = F.pairwise_distance(z1,z2,keepdim=True)
        loss = torch.mean((1-label)*torch.pow(dist,2) + label * torch.pow(torch.clamp((self.margin - dist), min=0),2))
        acc = ((dist > 0.6) == label).float().mean()

        return loss, acc


class SiameseNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            convBlock(1,4),
            convBlock(4,8),
            convBlock(8,8),
            nn.Flatten(),
            nn.Linear(8*100*100, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x1):
        out1 = self.features(x1)

        return out1