from torch.utils.data import Dataset
from torchvision import transforms as T  
from torchvision.datasets.utils import download_url 

from PIL import Image 
import matplotlib.pyplot as plt
import os 
from pathlib import Path

DATA_DIR  = Path("data") 

transforms = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

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
        self.classes = list(set(parse_breed(fname) for fname in files))
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)} 
        self.transform = transform  
       
    def __len__(self): 
        return len(self.files) 
    
    def __getitem__(self, index):
        fname = self.files[index] 
        img = self.transform(open_image(fname)) 
        class_idx = self.class_to_idx[parse_breed(fname)]
        return img, class_idx 


dataset = PetsDataset(DATA_DIR, transforms) 
print(len(dataset))

