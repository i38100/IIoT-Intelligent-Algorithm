from PIL import Image
from torch.utils.data import Dataset
import os

class MyDataset(Dataset):
    def __init__(self, img_dir, img_prefix, label_path, transform=None):
        self.img_dir = img_dir
        self.img_prefix = img_prefix
        self.label_path = label_path
        self.transform = transform
    
    def __len__(self):
        imgs = os.listdir(self.img_dir)
        return len(imgs)
    
    def __getitem__(self, idx):
        file = str(idx)+'.jpg'
        img = Image.open(self.img_dir+self.img_prefix+file)
        with open(self.label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        label = int(labels[idx].strip().split(',')[1])
        return self.transform(img), label