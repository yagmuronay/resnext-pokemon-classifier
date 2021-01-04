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

"""### Let's define our PokeMon dataset
- Put the "pokemon" folder to somewhere of your codebase, and define the path to "data_path"
"""

data_path = './Dataset/pokemon'

class PokemonDataset(Dataset):
    def __init__(self, data_path, is_training):
        self.data_path = data_path
        self.train_path = os.path.join(data_path, 'train')
        self.test_path = os.path.join(data_path, 'validate')
        self.is_training = is_training
        if self.is_training:
            self.target_path = self.train_path
        else:
            self.target_path = self.test_path

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
        self.transform = torchvision.transforms.Compose([self.tensor_transform,
                                                         self.normalize_transform])

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        (fp, class_name, class_label) = self.img_path_label[idx]
        img = Image.open(fp)
        original_img = self.tensor_transform(img)
        input = self.transform(img)
        
        sample = dict()
        sample['input'] = input
        sample['original_img'] = original_img
        sample['target'] = class_label
        sample['class_name'] = class_name

        return sample

"""### Create dataset/dataloader for test """

test_dataset = PokemonDataset(data_path, False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

"""### Try to sample out one test dataset"""

sample = next(iter(test_dataloader))

plot_idx = 0
plt.imshow(sample['original_img'][plot_idx].permute(1, 2, 0))
plt.title(sample['class_name'][plot_idx])


"""### Choose your device - use GPU or not?"""

# device = 'cpu'
device = 'cuda'
print('Current Device : {}'.format(device))


"""### Load the Saved CheckPoint"""

# Code referred from: https://discuss.pytorch.org/t/saving-customized-model-architecture/21512/2
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model

model_path = './drive/MyDrive/Codes/models/pokemon_recent.pth' #./drive/MyDrive/Path/To/Save/Your/Model
model = load_checkpoint(model_path)


"""### Define a function for test"""

def test(model, sample):
    model.eval()

    with torch.no_grad():
        input = sample['input'].float().to(device)
        target = sample['target'].long().to(device) 

        pred = model(input)

        top3_val, top3_idx = torch.topk(pred, 3)

        num_correct = torch.sum(top3_idx == target.view(-1, 1))

    return num_correct.item()


"""### Run Test"""

### Validation Phase
# Initialize Loss and Accuracy
test_accu = 0.0

# Iterate over the val_dataloader
with tqdm(total=len(test_dataloader)) as pbar:
    for idx, sample in enumerate(test_dataloader):
        num_correct = test(model, sample)
        test_accu += num_correct / len(test_dataloader)
        pbar.update(1)

print('Total Accuracy: ', test_accu)