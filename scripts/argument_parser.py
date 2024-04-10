from argparse import ArgumentParser, BooleanOptionalAction


class Parser:
    def __init__(self, desc):
        self.args = ArgumentParser(description=desc)
    
    def add_creating_dataset_group(self):
        self.args.add_argument("--use_colab", action=BooleanOptionalAction, default=False, help="Булевый флаг, используется ли скрипт в Google Colab")
        self.args.add_argument("--train_size", type=float, default=0.9, help="Доля изображений, которая попадет в train выборку; остальные попадут в test", required=False)
        self.args.add_argument("--download_path", default=".", help="Путь, по которому будет скачан датасет", required=False)
        self.args.add_argument("--train_test_images_path", default=".", help="Путь, по которому будут созданы папки для train и test изображений", required=False)