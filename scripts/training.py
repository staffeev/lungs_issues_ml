from .argument_parser import Parser, get_class_from_file
import sys
sys.path.append("..")
import os
from core.preprocessing import get_train_transofrms, get_test_transforms
from core.architecture import test_architecture
from core.custom_dataset import CustomDataset
from torch import nn
from torch import optim

parser = Parser(desc="Обучение модели")
parser.add_training_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    dataset_data_path = ""
    img_path = os.path.join("dataset", "data", "train_images")
    dataset_train = CustomDataset(img_path, os.path.join("dataset", "data", "train_labels.csv"), 
                                  get_train_transofrms())
    dataset_test = CustomDataset(img_path, os.path.join("dataset", "data", "test_labels.csv"), get_test_transforms())
    test_architecture(
        dataset_train, dataset_test, get_class_from_file(args.model_path)(), eval(f"optim.{args.optimiser}"),
        eval(f"nn.{args.loss_func}()"), args.num_epochs, args.batch_size, args.logging_iters_train,
        args.logging_iters_valid, args.model_title, args.save_graph, args.save_state, args.load_state
    )