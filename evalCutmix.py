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

ckpt_path ='../FEWshot/cutmix/model_999.pth'
save_imgpath ='../FEWshot/cutmix/'
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
print(f"train_loss:{train_loss[-1]}")
print(f"val_loss:{val_loss[-1]}")
print(f"train_acc:{train_acc[-1]}")
print(f"val_acc:{val_acc[-1]}")
'''
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

'''
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

'''
class Face_Dataset():
    def __init__(self, data_dir, transformer=None):
        self.person_items = []
        for (root,dirs, files) in os.walk(data_dir):
            if len(files) > 0:
                for file_name in files:
                    self.person_items.append(os.path.join(root,file_name))

        self.transformer = transformer

    def __len__(self):
        return len(self.person_items)

    def __getitem__(self, idx):
        face_path = self.person_items[idx]
        face_image = cv2.imread(face_path,0)

        if self.transformer:
            face_image = self.transformer(face_image)
        person_name = Path(face_path).parent.name
        return face_image, person_name



def build_transformer(image_size=100):

    transformer = transforms.Compose([ transforms.ToPILImage(),
                                       transforms.Resize((100,100)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5), (0.5))
                                       ])
    return transformer

transformer = build_transformer()
dataset = Face_Dataset(data_dir, transformer=transformer)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)


all_images= []
all_labels= []
all_embeds=[]

for idx, sample in enumerate(dataloader):
    image = sample[0]
    label = sample[1]

    with torch.no_grad():
        embed = model(image.to(DEVICE))

    embed - embed.detach().cpu().numpy()

    image = make_grid(image, normalize=True).permute(1,2,0)
    image = cv2.resize(np.array(image),dsize=(80,80), interpolation=cv2.INTER_NEAREST)

    all_images.append(image)
    all_labels.append(label)
    all_embeds.append(embed)

all_images = torch.tensor(np.moveaxis(np.stack(all_images, axis=0), 3, 1))
all_embeds = torch.tensor(np.stack(all_embeds, axis=0).squeeze(1))
#all_labels = np.stack(all_labels, axis=1).squeeze(0)
all_labels = np.concatenate(all_labels).tolist()

writer.add_embedding(all_embeds, label_img=all_images, metadata=all_labels)
writer.close()

'''