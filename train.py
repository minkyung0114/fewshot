import torch
import torch.nn as nn
import os
from common import *
from fewshot import SiameseNetwork, ContrastiveLoss, build_dataloader

dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)
SAVE_DIR ='../FEWshot/'

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
                codeA, codeB = model(imgA, imgB)

            loss, acc = criterion(codeA, codeB, label)

            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

        losses[phase] = running_loss / len(dataloaders[phase])
        accuracies[phase] = running_acc / len(dataloaders[phase])

    return losses, accuracies




is_cuda = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')

model = SiameseNetwork()
model = model.to(DEVICE)
criterion = ContrastiveLoss(margin=2.0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 100

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

    if (epoch+1) % 5 == 0:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc }, os.path.join(SAVE_DIR, f"model_{epoch+1}.pth")
                   )





