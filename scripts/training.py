from .argument_parser import Parser, get_class_from_file
import sys
sys.path.append("..")
import os
from core.preprocessing import get_train_transofrms, get_test_transforms
from core.architecture import train_model
from core.custom_dataset import CustomDataset
from torch import nn
from torch import optim
import torch

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

parser = Parser(desc="Обучение модели")
parser.add_training_group()
parser.add_augmentation_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    if args.use_gpu:
        torch.set_default_device("cuda")
    img_path = os.path.join("dataset", "data", "train_images_masked" if args.mask else "train_images")
    augmentation_args = (args.resize, args.brightness, args.contrast, args.sharpness, args.equalize, args.invert)
    dataset_train = CustomDataset(img_path, os.path.join("dataset", "data", "train_labels.csv"), 
                                  get_train_transofrms(args.horflip, args.rotate), *augmentation_args)
    dataset_test = CustomDataset(img_path, os.path.join("dataset", "data", "test_labels.csv"), get_test_transforms(),
                                 *augmentation_args)
    train_model(
        dataset_train, dataset_test, get_class_from_file(args.model_path)(), eval(f"optim.{args.optimiser}"),
        eval(f"nn.{args.loss_func}()"), args.num_epochs, args.batch_size, args.logging_iters_train,
        args.logging_iters_valid, args.model_title, args.save_graph, args.save_state, args.load_state,
        args.period_save_weights
    )