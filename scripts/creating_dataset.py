import opendatasets as od
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from argument_parser import Parser


def create_dataset(flag_to_download=False, train_size=0.9):
    if flag_to_download:
        od.download("https://www.kaggle.com/competitions/ml-intensive-yandex-academy-spring-2024", data_dir="dataset")
    dataset_path = os.path.join("dataset", "data")
    ans = pd.read_csv(os.path.join(dataset_path, "train_answers.csv"))
    ans["id"] = ans["id"].apply(lambda x: f"img_{x}.png")
    train_ix, test_ix = train_test_split(ans, train_size=train_size)
    train_ix.to_csv(os.path.join(dataset_path, "train_labels.csv"), index=False)
    test_ix.to_csv(os.path.join(dataset_path, "test_labels.csv"), index=False)


parser = Parser(desc="Скачивание датасета и разделение изображений на train и test")
parser.add_creating_dataset_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    create_dataset(args.download, args.train_size)



