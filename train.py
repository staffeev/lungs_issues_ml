import json
from core import *
from models import *
from torch.nn import Module
from torch.optim.optimizer import Optimizer

def read_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return json.load(f)
    

def init_model(name: str, init: dict) -> Module:
    return eval(name)(**init)


def init_optimizer(model: Module, name: str, args: dict) -> Optimizer:
    return eval(name)(model.parameters(), **args)


def init_loss(loss_function_name: str) -> Module:
    return eval(loss_function_name)()


def kwargs_to_str(**kwargs):  # по приколу написанная функция. Не несёт никакого смысловой нагрузки.
    return ', '.join('='.join(map(str, i)) for i in kwargs.items())


def main():
    config = read_config('run_configs/simple_cnn.json')

    model = init_model(**config.pop('model'))
    optim = init_optimizer(model, **config.pop('optimizer'))
    metrics = get_metrics(config.pop('metrics'))
    loss = init_loss(config.pop('loss_function_name'))
    print('типо вызов: train_model(', model, metrics, 
          loss, optim, kwargs_to_str(**config), ')', sep='\n')


if __name__ == "__main__":
    main()

    
