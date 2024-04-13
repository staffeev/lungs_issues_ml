from .argument_parser import Parser, get_class_from_file
import sys
sys.path.append("..")
import torch
import os
from core.architecture import load_model_state
from core.preprocessing import get_test_transforms
from core.global_dataset import PatientCustomDataset
import csv
from tqdm import tqdm

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_predicts(dataset, healthy_model, coronavirus_model):
    data = torch.utils.data.DataLoader(dataset, shuffle=False, generator=torch.Generator(device))
    predicts = []
    x = 0
    for im in tqdm(data, desc=f"Getting predictions"):
        im = im.to(device)
        y_pred = healthy_model(im).argmax(1).item()
        y_pred[y_pred == 1] = coronavirus_model(im).argmax(1).item()
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
    dataset = PatientCustomDataset(os.path.join("dataset", "data", "test_images"), transform=get_test_transforms())
    healthy_model = get_class_from_file(f'{args.model_path}_healthy')().to(device)
    coronavirus_model = get_class_from_file(f'{args.model_path}_coronavirus')().to(device)

    load_model_state(args.weights, healthy_model)
    load_model_state(args.weights, coronavirus_model)
    save_predicts(get_predicts(dataset, healthy_model, coronavirus_model), args.save_path)    