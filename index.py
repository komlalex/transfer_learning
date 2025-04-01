from torch.utils.data import Dataset
from torchvision import transforms as T  
from torchvision.datasets.utils import download_url 

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
plt.show() 
