from torchvision.transforms import transforms


def get_train_transofrms():
    return transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
