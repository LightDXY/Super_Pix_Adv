from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import scipy.io as scio
import torch

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class McDataset(Dataset):
    def __init__(self, image_dir, slic_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.slic_dir = slic_dir
        
        imgs= os.listdir(self.slic_dir)[:200]
        self.A_paths = []
        self.A_slics = []
        self.A_names = []
        for img in imgs :
            imgpath=os.path.join(self.slic_dir,img)
            self.A_slics.append(imgpath)
            self.A_paths.append(os.path.join(self.image_dir,img.split('.')[0]+'.png'))
            self.A_names.append(img.split('.')[0])
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        self.initialized = False
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        
        A_slic = self.A_slics[index % self.A_size]
        data = scio.loadmat(A_slic)
        A_slic = data['slic']
        A_index = torch.from_numpy(A_slic).float().repeat( 3, 1, 1).view( 3, -1).long()
        
        A_name = self.A_names[index % self.A_size]
        
        return {'A': A, 'path': A_path, 'index': A_index, 'name': A_name }
