import os
import shutil
import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from argument_parser import Parser
from tqdm import tqdm

slash = "/" if sys.platform == "linux" else "\\"


def move_images_to_classes(ixs, cur_path, new_path):
    ids, targets = ixs["id"].tolist(), ixs["target_feature"].tolist()
    images = [i for i in os.listdir(cur_path) if int(i.split("_")[1][:-4]) in ids]
    for c in tqdm(range(len(targets)), desc="Moving images"):
        image, target = images[c], targets[c]
        shutil.copyfile(rf"{cur_path}{slash}{image}", rf"{new_path}{slash}{target}{slash}{image}")


def create_folders(path):
    for f in ("train_data", "test_data"):
        os.system(f"mkdir {path}{slash}{f}")
        for class_num in (0, 1, 2):
            os.system(f"mkdir {path}{slash}{f}{slash}{class_num}")


def create_dataset(download_path=".", train_size=0.9, train_test_images_path="."):
    od.download("https://www.kaggle.com/competitions/ml-intensive-yandex-academy-spring-2024", data_dir=download_path)
    ans = pd.read_csv(f"{download_path}{slash}train_answers.csv")
    train_ix, test_ix = train_test_split(ans, train_size=train_size)
    create_folders(train_test_images_path)
    move_images_to_classes(train_ix, f"{download_path}{slash}train_images", f"{train_test_images_path}{slash}train_data")
    move_images_to_classes(test_ix, f"{download_path}{slash}train_images", f"{train_test_images_path}{slash}test_data")


parser = Parser(desc="Скачивание датасета и разделение изображений на train и test")
parser.add_creating_dataset_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    if args.use_colab:
        os.chdir("/content")
    create_dataset(args.download_path, args.train_size, args.train_test_images_path)



