from .argument_parser import Parser
import os
import sys
sys.path.append("..")
import importlib.util
from core.preprocessing import preprocessing_test, preprocessing_train
from core.architecture import test_architecture, load_model_state
from torch import nn
from torch import optim



parser = Parser(desc="Обучение модели")
parser.add_training_group()


if __name__ == "__main__":
    args = parser.args.parse_args()
    if args.use_colab:
        os.chdir("/content")
    dataset_train = preprocessing_train(args.train_data_path)
    dataset_test = preprocessing_test(args.test_data_path)

    spec = importlib.util.spec_from_file_location("module", args.model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_classname = [i for i in dir(module) if not i.startswith("_")][0]

    test_architecture(
        dataset_train, dataset_test, eval(f"module.{model_classname}()"), eval(f"optim.{args.optimiser}"),
        eval(f"nn.{args.loss_func}()"), args.num_epochs, args.batch_size, args.logging_iters_train,
        args.logging_iters_valid, args.model_title, args.save_graph, args.save_state, args.load_state
    )