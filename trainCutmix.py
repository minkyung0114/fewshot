import torch
import torch.nn as nn
import os
from common import *
from fewshot import SiameseNetwork, cutmix_ContrastiveLoss, build_dataloader,Face_Dataset,cutmix_Contrastiveacc,SiameseNetwork3,SiameseNetwork2
from cutmix import *
dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)
import matplotlib.pyplot as plt
from torch import optim
every_seed(42)

BETA=1.0
cutmix_prob = 0.5


SAVE_DIR_cutmix ='../FEWshot/cutmixAdam/'

dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)





'''
def save_model(model_state, model_name, save_dir=SAVE_DIR):

    os.makedirs(save_dir, exist_ok=True)
    #torch.save(model_state, os.path.join(save_dir, model_name))
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'train_acc':accs['train'],
                'val_acc':accs['val']

                },os.path.join(save_dir, model_name)
               )

'''




def train_one_epoch(dataloaders, model, criterion, optimizer, device):

    losses = {}
    accuracies = {}

    for phase in ["train", "val"]:
        running_loss = 0.0
        running_acc = 0.0

        if phase == "train":
            model.train()
        else:
            model.eval()


        for index, batch in enumerate(dataloaders[phase]):

            imgA = batch[0].to(device)
            imgB = batch[1].to(device)
            label = batch[2].to(device)

            with torch.set_grad_enabled(phase == "train"):


                r = np.random.rand(1)
                if BETA > 0 and r > cutmix_prob:

                    lam = np.random.beta(BETA, BETA)
                    rand_index = torch.randperm(imgB.size()[0]).to(device)
                    target_a = label
                    target_b = label[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(imgB.size(), lam)
                    imgB[:, :, bbx1:bbx2, bby1:bby2] = imgB[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgB.size()[-1] * imgB.size()[-2]))
                    #print(f"lam:{lam}")

                    codeA, codeB = model(imgA, imgB)

                    # loss, acc = criterion(codeA, codeB, label)
                    #loss = float(criterion(codeA, codeB, target_a)) * lam + float(criterion(codeA, codeB, target_b)) * (1. - lam)


                    '''
                    loss1 = criterion(codeA, codeB, target_a) * lam
                    loss2 = criterion(codeA, codeB, target_b) * (1-lam)
                    loss = loss1 + loss2
                    '''

                    loss1, acc1 = criterion(codeA, codeB, target_a)
                    loss1 = loss1 * lam
                    loss2, acc2 = criterion(codeA, codeB, target_b)
                    loss2 = loss2 * (1-lam)
                    loss = loss1 + loss2
                    acc = (acc1+acc2) / 2
                else:
                    codeA, codeB = model(imgA, imgB)
                    loss, acc = criterion(codeA, codeB, label)

                    '''
                    print(f"loss:{loss}")
                    print(f"loss1 type: {loss1.dtype}| loss1:{loss1} ")
                    print(f"loss2 type: {loss2.dtype}| loss1:{loss2} ")

                    print(f"acc:{acc}")
                    print(f"acc1 type: {acc1.dtype}| acc1:{acc1} ")
                    print(f"acc2 type: {acc2.dtype}| acc2:{acc2} ")
                    '''

                    #loss = criterion(codeA, codeB, target_a) * lam + criterion(codeA, codeB, target_b) * (1. - lam)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()

                running_loss += loss.item()
                running_acc += acc.item()

        losses[phase] = running_loss / len(dataloaders[phase])
        accuracies[phase] = running_acc / len(dataloaders[phase])

    return losses, accuracies




is_cuda = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')

model = SiameseNetwork2()
model = model.to(DEVICE)

criterion = cutmix_ContrastiveLoss(margin=2.0)
optimizer = optim.Adam(model.parameters(),lr = 0.0005 )
#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.99 ** epoch)
num_epochs = 1000

best_epoch = 0
best_score = 0.0
train_loss, train_acc = [], []
val_loss, val_acc = [], []

for epoch in range(num_epochs):
    losses, accs = train_one_epoch(dataloaders, model, criterion, optimizer, DEVICE)
    train_loss.append(losses["train"])
    val_loss.append(losses["val"])

    train_acc.append(accs["train"])
    val_acc.append(accs["val"])

    print(f"{epoch}/{num_epochs}| Train loss:{losses['train']:.4f}, Val loss :{losses['val']:.4f}, "
          f"|{epoch}/{num_epochs}| Train acc:{accs['train']:.4f}, Val acc :{accs['val']:.4f} ")

    '''if (epoch+1) % 5 == 0:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc }, os.path.join(SAVE_DIR_cutmix, f"model_{epoch}.pth")
                   )'''

torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc}, os.path.join(SAVE_DIR_cutmix, f"model_{epoch}.pth")
           )





