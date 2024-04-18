from core.segmentation_dataset import PatientCustomDataset
import os
from .argument_parser import Parser, get_class_from_file
import sys
sys.path.append("..")
from core.preprocessing import get_train_transofrms, get_test_transforms
from core.architecture import test_binary_architecture
from core.custom_dataset import CustomDataset
from torch import nn
from torch import optim
import torch

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


parser = Parser(desc="Обучение модели")
parser.add_training_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    if args.use_gpu:
        torch.set_default_device("cuda")
    img_path = os.path.join("dataset", "data", "train_images")
    healthy_dataset_train = CustomDataset(img_path, os.path.join("dataset", "data", "train_labels.csv"), 
                                                 get_train_transofrms(), True)
    healthy_dataset_test = CustomDataset(img_path, os.path.join("dataset", "data", "test_labels.csv"), 
                                                get_test_transforms(), True)
    healthy_model = get_class_from_file(args.model_path)()

    coronavirus_model = get_class_from_file(args.model_path)()
    print(args.optimiser)
    testy = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )
    testy.eval()
    
    testy()



