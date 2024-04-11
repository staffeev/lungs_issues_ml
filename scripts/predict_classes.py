from .argument_parser import Parser
import sys
sys.path.append("..")
import importlib.util
import torch
from torchvision.transforms import transforms
from core.architecture import load_model_state
from torch.utils.data import Dataset
import csv
import os
from PIL import Image
from tqdm import tqdm
import natsort


class CustomDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


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
    dataset = CustomDataset(args.images_path, transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Grayscale()
    ]))

    spec = importlib.util.spec_from_file_location("module", args.model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_classname = [i for i in dir(module) if not i.startswith("_")][0]
    model = eval(f"module.{model_classname}()")
    load_model_state(args.model_title, model)
    save_predicts(get_predicts(dataset, model), args.save_path)







