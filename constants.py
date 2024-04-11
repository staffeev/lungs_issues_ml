import sys

slash = "/" if sys.platform == "linux" else "\\"
dataset_name = "ml-intensive-yandex-academy-spring-2024"
dataset_data_path = f"{dataset_name}{slash}data{slash}"