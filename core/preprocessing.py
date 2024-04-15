from torchvision.transforms import transforms
import torch
from torchvision.transforms.v2 import functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_transofrms(flip_prob=0, rot_angle=0):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(flip_prob),
        transforms.RandomRotation(rot_angle),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
