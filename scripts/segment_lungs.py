from .argument_parser import Parser, get_class_from_file
import sys
sys.path.append("..")
import torch
from torchvision.transforms import ToPILImage
import os
from core.architecture import load_model_state
from core.preprocessing import get_test_transforms
from core.segmentation_dataset import SegmentationDataset
import csv
from tqdm import tqdm

# torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_masks(path, dataset, model):
    transform = ToPILImage()
    for image, name in dataset:
        image = image.to(device)
        result = transform(model(image))
        result.save(os.path.join(path, name), model)
        



parser = Parser(desc="Получения сегментаций для изображений")
parser.add_get_answers_group()
parser.add_augmentation_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    augmentation_args = (args.resize, args.brightness, args.contrast, args.sharpness, args.equalize, args.invert)
    dataset = SegmentationDataset(get_test_transforms(), os.path.join("dataset", "data"), 'test_images', 'test_images_mask',)
    model = get_class_from_file(args.model_path)().to(device)
    load_model_state(args.weights, model)
    save_masks(os.path.join("dataset", "data", 'test_images_mask'), dataset, model)
