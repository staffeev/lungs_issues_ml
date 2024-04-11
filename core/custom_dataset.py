import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(annotations_file) if annotations_file is not None else None

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.img_labels is None:
            img_path = os.path.join(self.img_dir, f"img_{idx}.png")
        else:
            img_path = os.path.join(self.img_dir, f"img_{self.img_labels.iloc[idx][0]}.png")
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.img_labels is None:
            return image
        return image, self.img_labels.iloc[idx][1]