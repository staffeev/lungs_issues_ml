from argparse import ArgumentParser, BooleanOptionalAction
import importlib.util
import os


def get_class_from_file(path):
    """Получение класса из файла, заданного пуием к нему"""
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_classname = [i for i in dir(module) if not i.startswith("_") and i[0].isupper()][0]
    return eval(f"module.{model_classname}")


class Parser:
    def __init__(self, desc):
        self.args = ArgumentParser(description=desc)
        self.args.add_argument("--use_gpu", default=False, action=BooleanOptionalAction, help="Нужно ли использовать GPU при обучении модели")
    
    def add_creating_dataset_group(self):
        self.args.add_argument("--download", action=BooleanOptionalAction, default=False, help="Булевый флаг, нужно ли скачивать датасет заново")
        self.args.add_argument("--train_size", type=float, default=0.9, help="Доля изображений, которая попадет в train выборку; остальные попадут в test")

    def add_training_group(self):
        self.args.add_argument("--model_title", type=str, default="model", help="Название модели")
        self.args.add_argument("model_path", type=str, help="Путь к файлу с моделью")
        self.args.add_argument("--optimiser", choices=[
            'ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 
            'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam'],
            default="Adam")
        self.args.add_argument("--loss_func", choices=[
            "L1Loss", "MSELoss", "CrossEntropyLoss", "CTCLoss", "NLLLoss", "PoissonNLLLoss",
            "GaussianNLLLoss", "KLDivLoss", "BCELoss", "BCEWithLogitsLoss", "MarginRankingLoss",
            "HingeEmbeddingLoss", "MultiLabelMarginLoss", "HuberLoss", "SmoothL1Loss", "SoftMarginLoss",
            "MultiLabelSoftMarginLoss", "CosineEmbeddingLoss", "MultiMarginLoss", "TripletMarginLoss", "TripletMarginWithDistanceLoss"],
            default="CrossEntropyLoss")
        self.args.add_argument("--batch_size", default=64, type=int, help="Размер батча")
        self.args.add_argument("--num_epochs", default=3, type=int, help="Количество эпох для обучения")
        self.args.add_argument("--save_graph", default=False, action=BooleanOptionalAction, help="Надо ли сохранять графики обучения")
        self.args.add_argument("--save_state", default=False, action=BooleanOptionalAction, help="Надо ли сохранять параметры модели")
        self.args.add_argument("--logging_iters_train", default=10, type=int, help="Метрики каждого i-го батча в train заносятся на график") 
        self.args.add_argument("--logging_iters_valid", default=3, type=int, help="Метрики каждого i-го батча в test заносятся на график") 
        self.args.add_argument("--load_state", default=None, help="Название файла с весами, чьи сохраненные параметры будут использованы в модели")
        self.args.add_argument("--period_save_weights", default=1, type=int, help="Каждые n эпох веса модели будут сохраняться в файлы model_n.pt, model_2n.pt...")

    def add_get_answers_group(self):
        self.args.add_argument("model_path", help="Путь к файлу с моделью")
        self.args.add_argument("--weights", default="model_0", help="Название файла в папке model_states, весы из которого будет использован для получения предсказаний для test")
        self.args.add_argument("--save_path", default=os.path.join("outputs", "model.csv"), help="Путь, по которому будет сохранен файл с прадсказаниями")


