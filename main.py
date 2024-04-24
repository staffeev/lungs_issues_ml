import json
from core import *
from models import *
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, random_split


def read_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return json.load(f)
    

def init_model(name: str, init: dict) -> Module:
    return eval(name)(**init)


def init_optimizer(model: Module, name: str, args: dict) -> Optimizer:
    return eval(name)(model.parameters(), **args)


def init_loss(loss_function_name: str) -> Module:
    return eval(loss_function_name)()


def init_datasets(train_valid_config: dict, 
                  test_config: dict) -> tuple[Dataset, Dataset, Dataset]:
    train_transform = get_transform(train_valid_config['transform'])
    test_transform = get_transform(test_config['transform'])

    dataset = ImageDataset(**train_valid_config['init'], transform=train_transform)
    test_dataset = ImageDataset(**test_config['init'], transform=test_transform)
    train_dataset, valid_dataset = random_split(dataset, train_valid_config['train_valid_fractions'])

    return train_dataset, valid_dataset, test_dataset 



def main():
    config = read_config('run_configs/simple_cnn.json')

    model = init_model(**config.pop('model'))
    optim = init_optimizer(model, **config.pop('optimizer'))
    metrics = get_metrics(config.pop('metrics'))
    loss = init_loss(config.pop('loss_function_name'))
    train_dataset, valid_dataset, test_dataset = init_datasets(
        config.pop('train_valid_dataset'),
        config.pop('test_dataset')
    )


if __name__ == "__main__":
    main()

    
