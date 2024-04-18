from PIL import Image
import os
from tqdm import tqdm


# Функция для загрузки изображений и их масок
def load_images_and_masks(image_dir, mask_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"): 
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            # Загрузка изображения и маски с помощью PIL
            image = Image.open(image_path)  # Преобразуем изображение в RGB, так как маски - одноканальные
            mask = Image.open(mask_path)  # Преобразуем маску в одноканальное черно-белое изображение
            images.append(image)
            masks.append(mask)
    return images, masks

# Функция для применения маски к изображению
def apply_mask(image, mask):
    # Применение маски к изображению
    masked_image = Image.new("L", image.size)
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            mask_pixel = mask.getpixel((x, y))
            masked_pixel = int(pixel * (mask_pixel / 255))
            masked_image.putpixel((x, y), masked_pixel)
    return masked_image

# Папки с изображениями и масками
if __name__ == '__main__':
    image_folder = "dataset/data/train_images"
    mask_folder = "dataset/data/train_lung_masks"
    output_folder = "dataset/data/train_images_masked"

    # Загрузка изображений и их масок
    images, masks = load_images_and_masks(image_folder, mask_folder)

    # Применение масок к изображениям и сохранение результатов
    for image, mask in tqdm(zip(images, masks), desc="Apply masks to image"):
        masked_image = apply_mask(image, mask)
        output_path = os.path.join(output_folder, os.path.basename(image.filename))
        masked_image.save(output_path)