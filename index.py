import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from torchvision import models
from torchvision import transforms as T  
from torchvision.datasets.utils import download_url  
from torchvision.utils import make_grid

from PIL import Image 
import matplotlib.pyplot as plt
import os 
from pathlib import Path 
from tqdm.auto import tqdm 

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

"""Define accuracy function"""
def accuracy(outputs, labels): 
    preds = torch.argmax(outputs, dim=1) 
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) 


class ImageClassificationBase(nn.Module):  
    def training_step(self, batch): 
        images, labels = batch 
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss 
    
    def validation_step(self, batch): 
        images, labels = batch 
        out = self(images)        # Generate predictions 
        loss = F.cross_entropy(out, labels) # Calcuate the loss
        acc = accuracy(out, labels)     # Calculate accuracy
        return {"val_loss": loss, "val_acc": acc}  
    
    def validation_epoch_ends(self, outputs): 
        batch_losses = [x["val_loss"] for x in outputs] 
        epoch_loss = torch.stack(batch_losses).mean().item() 
        batch_accs = [x["val_acc"] for x in outputs] 
        epoch_acc = torch.stack(batch_accs).mean().item() 
        return {"val_loss": epoch_loss, "val_acc": epoch_acc} 
    
    def epoch_ends(self, epoch, result): 
        print(f"\33[32m Epoch: {epoch} | train_loss: {result["train_loss"]:.4f} | val_loss: {result["val_loss"]:.4f} | val_acc: {result["val_acc"]:.4f}")


class PetsModel(ImageClassificationBase): 
    def __init__(self, num_classes, pretrained=True):
        super().__init__() 
        # Use a pretrained model 
        self.network = models.resnet34(pretrained=pretrained) 
        # Replace last layer 
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes) 

    def forward(self, xb): 
        return self.network(xb) 
    

"""GPU Utilities and Training Loop""" 
def get_default_device(): 
    """Pick GPU if available, else CPU""" 
    if torch.cuda.is_available(): 
        return torch.device("cuda")  
    else:
        return torch.device("cpu") 

def to_device(data, device): 
    if isinstance(data, (list, tuple)): 
        return [to_device(x, device) for x in data] 
    return data.to(device) 

class DeviceDataLoader(): 
    def __init__(self, dl, device):
        self.dl = dl 
        self.device = device 

    def __iter__(self): 
        for batch in self.dl: 
            yield to_device(batch, self.device) 
    
    def __len__(self): 
        return len(self.dl) 
    

def evalaute(model: nn.Module, val_dl): 
    model.eval() 
    with torch.inference_mode(): 
        outputs = [model.validation_step(batch) for batch in val_dl] 
        return model.validation_epoch_ends(outputs)  
    
def fit(epochs ,lr, model: nn.Module, train_dl, val_dl, opt_fuc = torch.optim.SGD): 
    history = [] 
    optimizer = opt_fuc(model.parameters(), lr) 
    for epoch in range(epochs): 
        # Training Phase 
        model.train() 
        train_losses = [] 
        for batch in tqdm(train_dl): 
            loss = model.training_step(batch) 
            train_losses.append(loss)  
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 

        # Validation Phase 
        result = evalaute(model, val_dl) 
        result["train_loss"] = torch.stack(train_losses).mean().item() 
        model.epoch_ends(epoch, result) 
        history.append(result) 
    return history 

def get_lr(optimizer: torch.optim.Optimizer): 
    for param_group in optimizer.param_groups: 
        return param_group["lr"] 
    
def fit_one_cycle(epochs: int, max_lr: float, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, 
                  weight_decay=0, grad_clip=None, opt_func= torch.optim.SGD): 
    torch.cuda.empty_cache() 
    history = [] 

    # Set up custom optimizer with weight decay 
    optimizer = opt_func(model.parameters(), weight_decay=weight_decay) 

    # Set up on-cycle learning rate scheduler 
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                epochs=epochs,
                                                max_lr=max_lr, 
                                                steps_per_epoch=len(train_dl)) 
    
    for epoch in range(epochs): 
        # Training Phase 
        model.train() 
        train_losses = [] 
        lrs = [] 
        for batch in tqdm(train_dl): 
            loss = model.training_step(batch) 
            train_losses.append(loss) 
            loss.backward() 
            
            # Gradient clipping 
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip) 

            optimizer.step() 
            optimizer.zero_grad() 

            # Record & update learning rate 
            lrs.append(get_lr(optimizer)) 
            sched.step()  


        # Validation Phase 
        result = evalaute(model, val_dl) 
        result["train_loss"] = torch.stack(train_losses).mean().item() 
        result["lrs"] = lrs  
        model.epoch_ends(epoch, result)
        history.append(result) 
    return history 

device = get_default_device() 

train_dl = DeviceDataLoader(train_dl, device) 
val_dl = DeviceDataLoader(val_dl, device)  

"""Finetuning the Pretrained Model""" 
model = PetsModel(len(dataset.classes)) 
to_device(model, device)
history = evalaute(model, val_dl)
print(history)

epochs = 6 
max_lr = 0.01 
grad_clip = 0.1 
weight_decay = 1e-4 
opt_func = torch.optim.Adam 

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                        grad_clip=grad_clip, 
                        weight_decay=weight_decay, 
                        opt_func=opt_func)

"""Training the Model from Scratch 
Let's repeat the training without using the weights from the pretrained ResNet34 model
""" 

model2 = PetsModel(len(dataset.classes), pretrained=False)
to_device(model2, device) 
history2 = evalaute(model2, val_dl) 

history += [fit_one_cycle(epochs, max_lr, model2, train_dl, val_dl, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func) 
] 

