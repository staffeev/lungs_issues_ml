import torch
from torchvision.transforms import transforms
from torchvision import datasets


def preprocessing_dataset(train_path="train_data", test_path="test_data"):
    transform_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale()
    ])
    train_images = datasets.ImageFolder(root=train_path, transform=transform_list)
    test_images = datasets.ImageFolder(root=train_path, transform=transform_list)
    return train_images, test_images