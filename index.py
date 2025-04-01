import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms as T  
from torchvision.datasets.utils import download_url  
from torchvision.utils import make_grid

from PIL import Image 
import matplotlib.pyplot as plt
import os 
from pathlib import Path

DATA_DIR  = Path("data") 


"""Load all the image paths in the directory"""
files = list(DATA_DIR.glob("./*/*.jpg"))
#print(len(files))
#print(files[4]) 

"""Get the breed of the image from it's path name"""
def parse_breed(fname): 
    fname = os.path.split(fname)[-1]
    parts = fname.split("_") 
    return " ".join(parts[:-1])  

#print(parse_breed(files[5000]))  
#classes = list(set(parse_breed(fname) for fname in files)) 
#print(len(classes))


"""Open the image"""
def open_image(path): 
    with open(path, "rb") as f: 
        img = Image.open(f) 
        return img.convert("RGB")

"""View image""" 
plt.figure(figsize=(10, 15))
plt.imshow(open_image(files[5000]))
plt.title(parse_breed(files[5000]))
plt.axis(False)

"""Create a Custom PyTorch Dataset""" 
class PetsDataset(Dataset): 
    def __init__(self, root, transform):
        super().__init__() 
        self.root = root 
        self.files = list(root.glob("./*/*.jpg")) 
        self.classes = sorted(list(set(parse_breed(fname) for fname in files)))
        self.class_idx = {label: index for index, label in enumerate(self.classes)} 
        self.transform = transform  
       
    def __len__(self): 
        return len(self.files) 
    
    def __getitem__(self, index):
        fname = self.files[index] 
        img = self.transform(open_image(fname)) 
        class_idx = self.class_idx[parse_breed(fname)]
        return img, class_idx 

img_size = 224 
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
tfms = T.Compose([
    T.Resize(img_size),
    T.Pad(8, padding_mode="reflect"), 
    T.RandomCrop(img_size),
    T.ToTensor(), 
    T.Normalize(*imagenet_stats)
])

dataset = PetsDataset(DATA_DIR, tfms) 
classes = dataset.classes
def denormalize(images: torch.Tensor, means=imagenet_stats[0], stds=imagenet_stats[1]): 
    if len(images.shape) == 3:
        images = images.unsqueeze(0) 
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1) 
    return images * stds + means  

def show_image(img_tensor, label): 
    img_tensor = denormalize(img_tensor)[0].permute((1, 2, 0)) 
    plt.figure(figsize=(10, 15))
    plt.imshow(img_tensor) 
    plt.title(f"Label: {classes[label]}") 

#show_image(*dataset[5000]) 

"""Create training and validation datasets"""
BATCH_SIZE = 256
val_pct = 0.1 
val_size = int(val_pct * len(dataset)) 
train_size = len(dataset) - val_size 

train_ds, val_ds = random_split(dataset, [train_size, val_size]) 
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, pin_memory=True) 

def show_batch(batch): 
    images, _ = batch
    plt.figure(figsize=(16, 16))
    images = denormalize(images[:64]) 
    plt.imshow(make_grid(images, nrow=8).permute(1, 2, 0)) 
    plt.axis(False)

for batch in train_dl: 
    show_batch(batch) 
    break 

plt.show()

