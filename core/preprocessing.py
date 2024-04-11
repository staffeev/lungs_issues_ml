import torch
from torchvision.transforms import transforms
from torchvision import datasets



def preprocessing_train(train_path="train_data"):
    transform_list = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    return datasets.ImageFolder(root=train_path, transform=transform_list)


def preprocessing_test(test_path="test_data"):
    transform_list = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    return datasets.ImageFolder(root=test_path, transform=transform_list)