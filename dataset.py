import os
import zipfile
from torchvision import datasets, transforms


def unzip_data(zip_path="Data.zip", extract_to="Data"):

    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Data extracted to {extract_to}")
    else:
        print(f"Data already available in {extract_to}")


def get_dataset(data_root="Data", train=True):

    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    return dataset
