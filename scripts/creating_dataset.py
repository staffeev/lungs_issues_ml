import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
from argument_parser import Parser
from constants import slash, dataset_name, dataset_data_path


def create_dataset(flag_to_download=False, download_path=".", train_size=0.9, train_test_labels_path="."):
    if flag_to_download:
        od.download(f"https://www.kaggle.com/competitions/{dataset_name}", data_dir=download_path)
    dataset_path = f"{download_path}{slash}{dataset_data_path}"
    ans = pd.read_csv(f"{dataset_path}train_answers.csv")
    train_ix, test_ix = train_test_split(ans, train_size=train_size)
    train_ix.to_csv(f"{train_test_labels_path}{slash}train_labels.csv")
    test_ix.to_csv(f"{train_test_labels_path}{slash}test_labels.csv")
    return train_test_labels_path


parser = Parser(desc="Скачивание датасета и разделение изображений на train и test")
parser.add_creating_dataset_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    create_dataset(args.download, args.download_path, args.train_size, args.train_test_labels_path)



