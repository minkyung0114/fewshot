import cv2
from common import *
import torch
import os
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fewshot import  SiameseNetwork1, Face_Dataset,SiameseNetwork
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

ckpt_path ='../FEWshot/model_1000.pth'
save_imgpath ='../FEWshot/'
#print(f"ckpt_path:{ckpt_path}")
is_cuda = False
DEVICE = torch.device('cpu')
writer = SummaryWriter(logdir='runs/face_features')
#print(f"writer:{writer}")

def load_model(ckpt,device):

    checkpoint = torch.load(ckpt)
    model = SiameseNetwork()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)

checkpoint = torch.load(ckpt_path)
#print(f"checkpoint:{checkpoint}")
model = load_model(ckpt_path, DEVICE)
train_loss = checkpoint['train_loss']
#print(f"train loss:{train_loss}")
val_loss = checkpoint['val_loss']
train_acc = checkpoint['train_acc']
val_acc = checkpoint['val_acc']
epoch =checkpoint['epoch']
#print(f"epoch:{epoch}")
print(f"train_acc:{train_acc[-1]}")
print(f"val_acc:{val_acc[-1]}")
print(f"train_loss:{train_loss[-1]}")
print(f"val_loss:{val_loss[-1]}")

plt.figure(figsize=(5,5))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss")
plt.plot(train_loss,color = '#1f27b4',label='train_loss')
plt.plot(val_loss,color = '#ff1f0e',label='val_loss')
plt.legend()
plt.savefig(os.path.join(save_imgpath,f"fewshot_loss.png"),dpi=100)


plt.figure(figsize=(5,5))
plt.title('accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.plot(train_acc, color = '#2ca02c',label='train_acc')
plt.plot(val_acc, color = '#d69728',label='val_acc')
plt.legend()
plt.savefig(os.path.join(save_imgpath,f"fewshot_acc.png"),dpi=100)
plt.show()



#print(model)
transformer = transforms.Compose([ transforms.ToPILImage(),
                                   transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))
                                   ])


index =15

@torch.no_grad()
def recognize_face(dataset, index, model, device):

    val_dataset = dataset(data_dir, phase="val")
    #print(val_dataset.__getitem__(0))
    faceA,faceB,label = val_dataset[index]
    tensor_faceA = transformer(faceA).to(device) # C,H,W (1,100,100)
    tensor_faceB = transformer(faceB).to(device)

    codeA, codeB = model(tensor_faceA.unsqueeze(0), tensor_faceB.unsqueeze(0)) #B,C,H,W (1,1,100,100)
    euclidean_distance =F.pairwise_distance(codeA, codeB)

    output = "Same person" if euclidean_distance.item() < 0.6 else "Different person"
    return faceA, faceB, euclidean_distance.item(), output


'''
index=8
faceA, faceB, distance, output = recognize_face(Face_Dataset, index, model, device)
plt.figure(figsize=(7, 4))
plt.suptitle(f"{output} - Dissimilarity: {distance:.2f}")
plt.subplot(121)
plt.imshow(faceA, cmap='gray')
plt.subplot(122)
plt.imshow(faceB, cmap='gray')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,10))
'''
'''

for i in range(5):
    faceA, faceB, distance, output = recognize_face(Face_Dataset, index, model, DEVICE)
    plt.figure(figsize=(7, 4))
    plt.suptitle(f"{output} - Dissimilarity: {distance:.2f}")
    plt.subplot(121)
    plt.title("sample_imgA")
    plt.imshow(faceA, cmap='gray')
    plt.subplot(122)
    plt.title("sample_imgB")
    plt.imshow(faceB, cmap='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(save_imgpath,f'output_{i}.png'), dpi=100)
    plt.show()


'''



for i in range(5):
    faceA, faceB, distance, output = recognize_face(Face_Dataset, index, model, DEVICE)
    plt.figure(figsize=(7, 4))
    plt.suptitle(f"{output} - Dissimilarity: {distance:.2f}")
    plt.subplot(121)
    plt.title("sample_imgA")
    plt.imshow(faceA, cmap='gray')
    plt.subplot(122)
    plt.title("sample_imgB")
    plt.imshow(faceB, cmap='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(save_imgpath,f'output_{i}.png'), dpi=100)
    plt.show()


