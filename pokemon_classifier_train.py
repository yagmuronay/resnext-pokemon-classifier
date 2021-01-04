from tensorflow import summary
import tensorflow as tf

import os
import csv
import numpy as np
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

# Module for Importing Images
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import Model

data_path = './Dataset/pokemon'  # your base path to data set
model_dir = './Codes/models'  # your path to save the model


class PokemonDataset(Dataset):
    def __init__(self, data_path, is_training):
        self.data_path = data_path
        self.train_path = os.path.join(data_path, 'train')
        self.val_path = os.path.join(data_path, 'validate')
        self.is_training = is_training
        if self.is_training:
            self.target_path = self.train_path
        else:
            self.target_path = self.val_path

        self.classes = sorted(os.listdir(self.target_path))
        self.img_path_label = list()

        for c in self.classes:
            img_list = os.listdir(os.path.join(self.target_path, c))
            for fp in img_list:
                full_fp = os.path.join(self.target_path, c, fp)
                self.img_path_label.append((full_fp, c, self.classes.index(c)))

        # Add some tranforms for data augmentation.
        self.tensor_transform = torchvision.transforms.ToTensor()
        self.normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
        self.random_crop = torchvision.transforms.RandomCrop(size=170)
        self.random_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.resize = torchvision.transforms.Resize(size=224)
        self.train_transform = torchvision.transforms.Compose([self.tensor_transform,
                                                               #    self.random_crop,
                                                               self.random_flip,
                                                               self.resize,
                                                               self.normalize_transform])
        self.validate_transform = torchvision.transforms.Compose([self.tensor_transform,
                                                                  self.normalize_transform])

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        (fp, class_name, class_label) = self.img_path_label[idx]
        img = Image.open(fp)
        original_img = self.tensor_transform(img)

        if self.is_training:
            input = self.train_transform(img)
        else:
            input = self.validate_transform(img)

        sample = dict()
        sample['input'] = input
        sample['original_img'] = original_img
        sample['target'] = class_label
        sample['class_name'] = class_name

        return sample


"""### Set DataSet and DataLoader"""

batch_size = 64

train_dataset = PokemonDataset(data_path, True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = PokemonDataset(data_path, False)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

num_classes = 18

"""### Take a sample and try to look at the one"""

sample = next(iter(train_dataloader))

fig, ax = plt.subplots(1, 7, figsize=(20, 10))
for i in range(7):
    ax[i].imshow(sample['input'][i].permute(1, 2, 0))
    ax[i].set_title(sample['class_name'][i])

"""### Choose your device - use GPU or not?"""

# device = 'cpu'
device = 'cuda'
print('Current Device : {}'.format(device))

model = Model()
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

model(sample['input'].to(device)).shape

"""### Define functions for train/validation"""


def train(model, optimizer, sample):
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    input = sample['input'].float().to(device)
    target = sample['target'].long().to(device)

    pred = model(input)
    pred_loss = criterion(pred, target)

    top3_val, top3_idx = torch.topk(pred, 3)

    num_correct = torch.sum(top3_idx == target.view(-1, 1))

    pred_loss.backward()

    optimizer.step()

    return pred_loss.item(), num_correct.item()


def validate(model, sample):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        input = sample['input'].float().to(device)
        target = sample['target'].long().to(device)

        pred = model(input)
        pred_loss = criterion(pred, target)

        top3_val, top3_idx = torch.topk(pred, 3)

        num_correct = torch.sum(top3_idx == target.view(-1, 1))

    return pred_loss.item(), num_correct.item()


"""### Prepare the Tensorboard"""

train_log_dir = './runs/train'
train_summary_writer = summary.create_file_writer(train_log_dir)
val_log_dir = './runs/validate'
val_summary_writer = summary.create_file_writer(val_log_dir)

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir runs

"""### Run Training"""

max_epoch = 200
save_stride = 10
tmp_path = './checkpoint.pth'
max_accu = -1
for epoch in tqdm(range(max_epoch)):
    ### Train Phase

    # Initialize Loss and Accuracy
    train_loss = 0.0
    train_accu = 0.0

    # Load the saved MODEL AND OPTIMIZER after evaluation.
    if epoch > 0:
        checkpoint = torch.load(tmp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # how about learning rate scheduler?

    # Iterate over the train_dataloader
    with tqdm(total=len(train_dataloader)) as pbar:
        for idx, sample in enumerate(train_dataloader):
            curr_loss, num_correct = train(model, optimizer, sample)
            train_loss += curr_loss / len(train_dataloader)
            train_accu += num_correct / len(train_dataset)
            pbar.update(1)

    # Write the current loss and accuracy to the Tensorboard
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=epoch)
        tf.summary.scalar('accuracy', train_accu, step=epoch)

        # save the model and optimizer's information before the evaulation
    checkpoint = {
        'model': Model(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Save the checkpoint - you can try to save the "best" model with the validation accuracy/loss
    torch.save(checkpoint, tmp_path)
    if (epoch + 1) % save_stride == 0:
        torch.save(checkpoint, os.path.join(model_dir, 'pokemon_{}.pth'.format(epoch + 1)))
    torch.save(checkpoint, os.path.join(model_dir, 'pokemon_recent.pth'))

    ### Validation Phase
    # Initialize Loss and Accuracy
    val_loss = 0.0
    val_accu = 0.0

    # Iterate over the val_dataloader
    with tqdm(total=len(val_dataloader)) as pbar:
        for idx, sample in enumerate(val_dataloader):
            curr_loss, num_correct = validate(model, sample)
            val_loss += curr_loss / len(val_dataloader)
            val_accu += num_correct / len(val_dataloader)
            pbar.update(1)

    # Write the current loss and accuracy to the Tensorboard
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss, step=epoch)
        tf.summary.scalar('accuracy', val_accu, step=epoch)

    max_accu = max(val_accu, max_accu)
    if max_accu == val_accu:
        # Save your best model to the checkpoint
        torch.save(checkpoint, os.path.join(model_dir, 'pokemon_best.pth'))

    print(train_accu, val_accu)
