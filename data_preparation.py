import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""
# Pytorch version

# MNIST
def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    # MNIST Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
    ])

    # Download MNIST Dataset
    full_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)
    # 전체 길이
    total_len = len(full_dataset)
    third_len = total_len // 3
    remainder = total_len % 3  # 데이터 개수가 3으로 안 나누어떨어질 경우

    # 나눌 크기 지정 (예: [1/3, 1/3, 1/3 + 나머지])
    lengths = [third_len, third_len, third_len + remainder]
    # 데이터셋 분할
    subset1, subset2, subset3 = random_split(full_dataset, lengths, generator=torch.Generator().manual_seed(42))
        
    full_dataset = subset2
    # Splitting the full dataset into train, validation, and test sets
    test_split = 0.2
    train_size = int((1 - validation_split - test_split) * len(full_dataset))
    validation_size = int(validation_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

    # DataLoader for training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
    ])

    # Load the test set of MNIST Dataset
    val_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)

    # DataLoader for validation
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return gl_val_loader