import zipfile
import os
from pathlib import Path
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

import torchvision
from torchvision import datasets, transforms

# Load environment variables from .env file
load_dotenv()

def unzip_file(zip_file_path=os.getenv("zip_file_path"), 
               extract_to=os.getenv("data_path")):
    """
    Unzips the specified zip file to the target directory.
    
    Args:
        zip_file_path (str): The path to the zip file.
        extract_to (str): The directory where files will be extracted.
    """
    datasets_folder = os.path.join(extract_to, "datasets")

    # Check if the datasets folder exists
    if not os.path.exists(datasets_folder):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Extract only files that are not in the __MACOSX directory
                for member in zip_ref.namelist():
                    if not member.startswith('__MACOSX/'):
                        zip_ref.extract(member, extract_to)
            print(f"[INFO] Extracted all files to {extract_to}")
        except FileNotFoundError:
            print(f"[ERROR] The file {zip_file_path} was not found.")
        except zipfile.BadZipFile:
            print(f"[ERROR] The file {zip_file_path} is not a valid zip file.")
    else:
        print(f"[INFO] The folder '{datasets_folder}' already exists. Skipping extraction.")

def create_dataloaders(train_dir: str,
                       val_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int= os.cpu_count()):
    """Creates training,testing and validation DataLoaders.

  Takes in a training, testing and validation directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    val_dir: Path to validation directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloaderm validation_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             val_dir=path/to/val_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
    """
    
    # Loads dataset

    train_data = datasets.ImageFolder(train_dir,transform=transform)
    val_dataset = datasets.ImageFolder(val_dir,transform=transform)
    
    # Splitting train data into train and test
    test_size = int(0.2 * len(train_data))
    train_size = len(train_data) - test_size

    # Randomly split the dataset into train and test subsets
    train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
    
    class_names = train_data.classes

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,) # Refer to [https://horace.io/brrr_intro.html]
    
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    
    return train_dataloader,test_dataloader, val_dataloader,class_names


if __name__ == "__main__":
    unzip_file()