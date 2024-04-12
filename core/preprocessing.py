from torchvision.transforms import transforms
import torch

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_transofrms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
