import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings('ignore')

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


# Болен, не болен.
class PatientCustomDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, transform=None, healthy=True):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = None
        if annotations_file:       
            self.img_labels = pd.read_csv(annotations_file)
            if healthy:  # делить здоровых/больных, т.е скастить не нули к 1.
                self.img_labels['target_feature'][self.img_labels['target_feature'] != 0] = 1
            else: # оставить только больных и искать корону. (1, 2)
                self.img_labels.drop(self.img_labels[self.img_labels['target_feature'] == 0].index, inplace=True)
                self.img_labels['target_feature'] -= 1
        

    def __len__(self):
        return len(self.img_labels) if self.img_labels is not None else len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.img_labels is None:
            img_path = os.path.join(self.img_dir, f"img_{idx}.png")
        else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx][0])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.img_labels is None:
            return image
        return image, self.img_labels.iloc[idx][1]