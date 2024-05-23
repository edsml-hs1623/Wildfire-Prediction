# data.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Global device parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# task 1 packaging
class ImageDataset(Dataset):
    """
    Custom Dataset class for handling image data.

    Args:
        data (numpy.ndarray): Input data array.

    Attributes:
        data (torch.Tensor): Processed data tensor.
    """
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Sample data tensor.
        """
        return self.data[idx]


def load_data(file_path):
    return np.load(file_path)


def preprocess_data(data, resample_rate=10):
    resampled_data = data[::resample_rate]
    reshaped_data = resampled_data[:, np.newaxis, :, :]
    return reshaped_data


def create_dataloader(data, batch_size=16, shuffle=True):
    dataset = ImageDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# task 2 packaging
class NumpyDataset(Dataset):
    """
    Custom Dataset class for handling image data stored in a NumPy file.

    Args:
        data_path (str): Path to the NumPy file containing the dataset.
        transform (callable, optional): Optional transform to be applied
            on a sample.

    Attributes:
        data (numpy.ndarray): Array containing the dataset.
        transform (callable, optional): Transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        self.data = np.load(data_path)
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Sample data tensor, optionally transformed.
        """
        image = self.data[idx].astype(np.float32)
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image


def data_loader(transform, train_path, test_path, bs=100):
    """
    Load training and testing datasets and create DataLoader objects for each.

    Parameters:
    - transform (callable): Transformations to apply to the data.
    - train_path (str): Path to the training dataset.
    - test_path (str): Path to the testing dataset.
    - bs (int, optional): Batch size for the DataLoader. Default is 100.

    Returns:
    - tuple: A tuple containing:
        - DataLoader: DataLoader for the training dataset.
        - DataLoader: DataLoader for the testing dataset.
    """
    train_data = NumpyDataset(train_path, transform=transform)
    test_data = NumpyDataset(test_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=True)

    return train_loader, test_loader
