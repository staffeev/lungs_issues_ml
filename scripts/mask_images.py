from PIL import Image
import os
from tqdm import tqdm


# Функция для применения маски к изображению
def apply_mask(image, mask):
    # Применение маски к изображению
    return Image.composite(image, mask, mask)
    

# Папки с изображениями и масками
if __name__ == '__main__':
    image_folder = "dataset/data/train_images"
    mask_folder = "dataset/data/train_lung_masks"
    output_folder = "dataset/data/train_images_masked"

    # Загрузка изображений и их масок
    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith(".png"): 
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)
            
            image = Image.open(image_path)  # Преобразуем изображение в RGB, так как маски - одноканальные
            mask = Image.open(mask_path)  # Преобразуем маску в одноканальное черно-белое изображение    for image, mask in tqdm(zip(images, masks), desc="Apply masks to image"):
            masked_image = apply_mask(image, mask)
            output_path = os.path.join(output_folder, os.path.basename(image.filename))
            masked_image.save(output_path)