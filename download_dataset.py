"""Скрипт, для простой загрузки датасета из kaggle."""
import opendatasets as od
import os


def main():
    od.download("https://www.kaggle.com/competitions/ml-intensive-yandex-academy-spring-2024")
    os.rename("ml-intensive-yandex-academy-spring-2024", "dataset")


if __name__ == "__main__":
    main()
