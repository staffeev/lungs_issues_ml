from .argument_parser import Parser, get_class_from_file
import sys
sys.path.append("..")
import torch
import os
from core.architecture import load_model_state
from core.preprocessing import get_test_transforms
from core.custom_dataset import CustomDataset
import csv
from tqdm import tqdm


def get_predicts(dataset, model):
    data = torch.utils.data.DataLoader(dataset, shuffle=False)
    predicts = []
    x = 0
    for im in tqdm(data, desc="Getting predictions"):
        y_pred = model(im).max(1)[1].item()
        predicts.append((x, y_pred))
        x += 1
    return predicts


def save_predicts(predicts, path):
    with open(path, "w", encoding="utf-8", newline="") as file:
        wr = csv.writer(file, delimiter=",")
        wr.writerow(["id", "target_feature"])
        wr.writerows(predicts)


parser = Parser(desc="Получения предсказаний для изображений")
parser.add_get_answers_group()

if __name__ == "__main__":
    args = parser.args.parse_args()
    dataset = CustomDataset(os.path.join("dataset", "path", "test_images"), transform=get_test_transforms())
    model = get_class_from_file(args.model_path)()
    load_model_state(args.model_title, model)
    save_predicts(get_predicts(dataset, model), args.save_path)







