from argparse import ArgumentParser, BooleanOptionalAction
import sys

slash = "/" if sys.platform == "linux" else "\\"


class Parser:
    def __init__(self, desc):
        self.args = ArgumentParser(description=desc)
        self.args.add_argument("--use_colab", action=BooleanOptionalAction, default=False, help="Булевый флаг, используется ли скрипт в Google Colab")
    
    def add_creating_dataset_group(self):
        self.args.add_argument("--download", action=BooleanOptionalAction, default=False, help="Булевый флаг, нужно ли скачивать датасет заново")
        self.args.add_argument("--train_size", type=float, default=0.9, help="Доля изображений, которая попадет в train выборку; остальные попадут в test", required=False)
        self.args.add_argument("--download_path", default=".", help="Путь, по которому находится скачанный датасет", required=False)
        self.args.add_argument("--train_test_images_path", default=".", help="Путь, по которому будут созданы папки для train и test изображений", required=False)

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
        self.args.add_argument("--load_state", default=None, help="Название модели, чьи сохраненные параметры будут использованы в модели")
        self.args.add_argument("--train_data_path", default="train_data", help="Путь с train изображениям")
        self.args.add_argument("--test_data_path", default="test_data", help="Путь к test изображениям")
    
    def add_get_answers_group(self):
        self.args.add_argument("model_path", help="Путь к файлу с моделью")
        self.args.add_argument("--model_title", default="model", help="Название модели, которая будет использована для получения предсказаний для test")
        self.args.add_argument("--images_path", default=f"ml-intensive-yandex-academy-spring-2024{slash}data{slash}test_images", 
                               help="Путь к папке с изображениями, для которых нужно получить предсказания")
        self.args.add_argument("--save_path", default="pred.csv", help="Путь, по которому будет сохранен файл с прадсказаниями")


