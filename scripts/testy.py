from core.global_dataset import PatientCustomDataset
import os
from .argument_parser import Parser, get_class_from_file
import sys
sys.path.append("..")
from core.preprocessing import get_train_transofrms, get_test_transforms
from core.architecture import test_binary_architecture
from core.global_dataset import PatientCustomDataset
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
    healthy_dataset_train = PatientCustomDataset(img_path, os.path.join("dataset", "data", "train_labels.csv"), 
                                                 get_train_transofrms(), True)
    healthy_dataset_test = PatientCustomDataset(img_path, os.path.join("dataset", "data", "test_labels.csv"), 
                                                get_test_transforms(), True)
    healthy_model = get_class_from_file(args.model_path)()

    coronavirus_dataset_train = PatientCustomDataset(img_path, os.path.join("dataset", "data", "train_labels.csv"), 
                                                     get_train_transofrms(), False)
    coronavirus_dataset_test = PatientCustomDataset(img_path, os.path.join("dataset", "data", "test_labels.csv"), 
                                                    get_test_transforms(), False)
    coronavirus_model = get_class_from_file(args.model_path)()
    print(args.optimiser)
    test_binary_architecture(
        healthy_dataset_train, healthy_dataset_test, healthy_model, eval(f"optim.{args.optimiser}"), eval(f"nn.{args.loss_func}()"), 
        coronavirus_dataset_train, coronavirus_dataset_test, coronavirus_model, eval(f"optim.{args.optimiser}"), eval(f"nn.{args.loss_func}()"), 

        args.num_epochs, args.batch_size, args.logging_iters_train,
        args.logging_iters_valid, args.model_title, args.save_graph, args.save_state, args.load_state,
        args.period_save_weights
    )

