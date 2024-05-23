import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error
import pytest


# DATALOADER
class NumpyDataset:
    """
    Dataset class to load numpy arrays as PyTorch tensors.

    Args:
        data_path (str): Path to the numpy data file.
        transform (callable, optional):
        Optional transform to be applied to the data.
    """

    def __init__(self, data_path, transform=None):
        """
        Initializes the NumpyDataset.

        Args:
            data_path (str): Path to the numpy data file.
            transform (callable, optional):
            Optional transform to be applied to the data.
        """

        self.data = np.load(data_path).astype(np.float32)
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            torch.Tensor: The item converted to a PyTorch tensor.
        """

        image = self.data[idx]
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image


class TestDataLoader:
    @pytest.fixture
    def dummy_data(self):
        """
        Fixture to generate dummy numpy arrays for training and testing.

        Returns:
            tuple:
            A tuple containing the dummy training and testing numpy arrays.
        """

        dummy_train = np.random.randn(200, 256, 256).astype(np.float32)
        dummy_test = np.random.randn(100, 256, 256).astype(np.float32)
        np.save('dummy_train.npy', dummy_train)
        np.save('dummy_test.npy', dummy_test)
        return dummy_train, dummy_test

    @pytest.fixture
    def dataloader(self, dummy_data):
        """
        Fixture to create training and testing DataLoaders using dummy data.

        Args:
            dummy_data (tuple):
            A tuple containing the dummy training and testing numpy arrays.

        Returns:
            tuple: A tuple containing the training and testing DataLoaders.
        """

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_data = NumpyDataset('dummy_train.npy', transform)
        test_data = NumpyDataset('dummy_test.npy', transform)
        train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
        return train_loader, test_loader

    def test_train_loader_batch_size(self, dataloader):
        """
        Test the batch size of the training DataLoader.

        Args:
            dataloader: Pytest fixture providing the training DataLoader.

        Raises:
            AssertionError:
            If the batch size of any batch does not match the expected value.
        """

        train_loader, _ = dataloader
        for i, batch in enumerate(train_loader):
            assert batch.shape[0] == 100, (
                f"Batch {i + 1}: Train loader batch size mismatch"
            )
            break  # Test only the first batch to keep the test fast

    def test_train_loader_batch_shape(self, dataloader):
        """
        Test the shape of batches in the training DataLoader.

        Args:
            dataloader: Pytest fixture providing the training DataLoader.

        Raises:
            AssertionError:
              If the shape of any batch does not match the expected shape.
        """

        train_loader, _ = dataloader
        for i, batch in enumerate(train_loader):
            assert batch.shape[1:] == (1, 256, 256), (
                f"Batch {i + 1}: Train loader batch shape mismatch"
            )
            break  # Test only the first batch to keep the test fast

    def test_train_loader_batch_type(self, dataloader):
        """
        Test the data type of batches in the training DataLoader.

        Args:
            dataloader: Pytest fixture providing the training DataLoader.

        Raises:
            AssertionError: If any batch is not of type torch.Tensor.
        """

        train_loader, _ = dataloader
        for i, batch in enumerate(train_loader):
            assert isinstance(batch, torch.Tensor), (
                f"Batch {i + 1}: Train loader batch is not a Tensor"
            )
            break  # Test only the first batch to keep the test fast

    def test_train_loader_batch_dtype(self, dataloader):
        """
        Test the data type of elements in batches in the training DataLoader.

        Args:
            dataloader: Pytest fixture providing the training DataLoader.

        Raises:
            AssertionError:
            If the data type of any batch does not match the expected data.
        """

        train_loader, _ = dataloader
        for i, batch in enumerate(train_loader):
            assert batch.dtype == torch.float32, (
                f"Batch {i + 1}: Train loader batch dtype mismatch"
            )
            break  # Test only the first batch to keep the test fast

    def test_test_loader_batch_size(self, dataloader):
        """
        Test the batch size of the test DataLoader.

        Args:
            dataloader: Pytest fixture providing the test DataLoader.

        Raises:
            AssertionError:
            If the batch size of any batch does not match the expected value.
        """

        _, test_loader = dataloader
        for i, batch in enumerate(test_loader):
            assert batch.shape[0] == 100, (
                f"Batch {i + 1}: Test loader batch size mismatch"
            )
            break  # Test only the first batch to keep the test fast

    def test_test_loader_batch_shape(self, dataloader):
        """
        Test the shape of batches in the test DataLoader.

        Args:
            dataloader: Pytest fixture providing the test DataLoader.

        Raises:
            AssertionError:
            If the shape of any batch does not match the expected shape.
        """

        _, test_loader = dataloader
        for i, batch in enumerate(test_loader):
            assert batch.shape[1:] == (1, 256, 256), (
                f"Batch {i + 1}: Test loader batch shape mismatch"
            )
            break  # Test only the first batch to keep the test fast

    def test_test_loader_batch_type(self, dataloader):
        """
        Test the data type of batches in the test DataLoader.

        Args:
            dataloader: Pytest fixture providing the test DataLoader.

        Raises:
            AssertionError: If any batch is not of type torch.Tensor.
        """

        _, test_loader = dataloader
        for i, batch in enumerate(test_loader):
            assert isinstance(batch, torch.Tensor), (
                f"Batch {i + 1}: Test loader batch is not a Tensor"
            )
            break  # Test only the first batch to keep the test fast

    def test_test_loader_batch_dtype(self, dataloader):
        """
        Test the data type of elements in batches in the test DataLoader.

        Args:
            dataloader: Pytest fixture providing the test DataLoader.

        Raises:
            AssertionError:
            If the data type of any batch does not match the expected data.
        """

        _, test_loader = dataloader
        for i, batch in enumerate(test_loader):
            assert batch.dtype == torch.float32, (
                f"Batch {i + 1}: Test loader batch dtype mismatch"
            )
            break  # Test only the first batch to keep the test fast


# GAN MODEL
class Generator(nn.Module):
    """Generator module for the GAN.

    This module generates images from random noise.

    Args:
        g_input_dim (int, optional): Dimension of the input noise vector.
            Defaults to 100.
        g_output_dim (int, optional): Dimension of the output image.
            Defaults to 256.

    Attributes:
        init_size (int): Initial size for reshaping the input noise vector.
        fc1 (nn.Linear): Fully connected layer for initial transformation
            of input noise.
        conv_blocks (nn.Sequential): Sequential convolutional layers
            for generating images.

    Methods:
        forward(x): Forward pass through the generator network.

    """

    def __init__(self, g_input_dim=100, g_output_dim=256):
        super().__init__()
        self.init_size = g_output_dim // 16
        self.fc1 = nn.Linear(g_input_dim, 128 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """Forward pass through the generator network.

        Args:
            x (torch.Tensor): Input noise vector.

        Returns:
            torch.Tensor: Generated images.
        """

        x = self.fc1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    """Discriminator module for the GAN.

    This module discriminates between real and fake images.

    Args:
        d_input_dim (int, optional): Dimension of the input image.
            Defaults to 256.

    Attributes:
        model (nn.Sequential): Sequential layers for image discrimination.
        linear_input_size (int): Size of the input to the linear layer.
        adv_layer (nn.Sequential): Sequential layers for adversarial
            classification.

    Methods:
        forward(x): Forward pass through the discriminator network.

    """

    def __init__(self, d_input_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        ds_size = d_input_dim // 8
        self.linear_input_size = 128 * ds_size ** 2
        self.adv_layer = nn.Sequential(
            nn.Linear(self.linear_input_size, 1), nn.Sigmoid())

    def forward(self, x):
        """Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor indicating the probability
                of the input being real.
        """

        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.adv_layer(x)
        return x


class TestGAN:
    """
    Test suite for evaluating the functionality of a Generative
    Adversarial Network (GAN).

    This class contains a series of test methods to validate different
    components and behaviors of a GAN, including the generator,
    discriminator, MSE calculation, and visualization of generated images.

    Attributes:
        batch_size (int): The batch size used for testing.
        g_input_dim (int): The input dimension for the generator.
        d_input_dim (int): The input dimension for the discriminator.
        transform (torchvision.transforms.Compose): A composition
            of image transformations.
        test_data (NumpyDataset): The test dataset for evaluating the GAN.
        test_loader (torch.utils.data.DataLoader): DataLoader for
            the test dataset.
    """

    @pytest.fixture
    def batch_size(self):
        """Fixture to provide the batch size."""
        return 16

    @pytest.fixture
    def g_input_dim(self):
        """Fixture to provide the input dimension for the generator."""
        return 100

    @pytest.fixture
    def d_input_dim(self):
        """Fixture to provide the input dimension for the discriminator."""
        return 256

    @pytest.fixture
    def generator(self, g_input_dim, d_input_dim):
        """Fixture to initialize the generator."""
        return Generator(g_input_dim=g_input_dim, g_output_dim=d_input_dim)

    @pytest.fixture
    def discriminator(self, d_input_dim):
        """Fixture to initialize the discriminator."""
        return Discriminator(d_input_dim=d_input_dim)

    @pytest.fixture
    def transform(self):
        """Fixture to provide transformations for the dataset."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    @pytest.fixture
    def test_data(self, transform, d_input_dim):
        """Fixture to provide the test dataset."""
        # Ensure that the test data dimensions match the input dimensions
        # of the discriminator
        return NumpyDataset('dummy_test.npy', transform)

    @pytest.fixture
    def test_loader(self, test_data, batch_size):
        """Fixture to provide the test data loader."""
        return DataLoader(test_data, batch_size, True)

    def visualize_generated_images(self, generator_model, z_dim=100,
                                   num_images_to_generate=5):
        """Generate and visualize images using the generator model.

        Args:
            generator_model (nn.Module): The generator model.
            z_dim (int, optional): Dimension of the input noise vector.
                Defaults to 100.
            num_images_to_generate (int, optional): Number of images
                to generate. Defaults to 5.

        Returns:
            torch.Tensor: Generated images.
        """

        generator_model.eval()
        with torch.no_grad():
            z = torch.randn(num_images_to_generate, z_dim)
            return generator_model(z)

    def test_generator(self, generator, batch_size, g_input_dim, d_input_dim):
        """Test the generator model.

        This function tests the output shape of the generator.

        Args:
            generator (Generator): The generator model.
            batch_size (int): Batch size.
            g_input_dim (int): Dimension of the input noise vector.
            d_input_dim (int): Dimension of the input image
                for the discriminator.
        """

        noise = torch.randn(batch_size, g_input_dim)
        generated_data = generator(noise)
        assert generated_data.shape == (batch_size, 1, d_input_dim,
                                        d_input_dim)

    def test_discriminator(self, discriminator, batch_size, d_input_dim):
        """Test the discriminator model.

        This function tests the output shape of the discriminator.

        Args:
            discriminator (Discriminator): The discriminator model.
            batch_size (int): Batch size.
            d_input_dim (int): Dimension of the input image
                for the discriminator.
        """

        images = torch.randn(batch_size, 1, d_input_dim, d_input_dim)
        output = discriminator(images)
        assert output.shape == (batch_size, 1)

    def test_mse_calculation(self, generator, test_loader):
        """Test the mean squared error (MSE) calculation.

        This function tests the calculation of MSE between generated
        and real images.

        Args:
            generator (Generator): The generator model.
            test_loader (DataLoader): DataLoader for the test dataset.
        """

        generator.eval()
        mse_loss = nn.MSELoss()
        total_mse = 0.0
        num_batches = 0

        with torch.no_grad():
            for images in test_loader:
                noise = torch.randn(images.size(0),
                                    generator.fc1.in_features)
                generated_images = generator(noise)

                mse = mse_loss(generated_images, images).item()
                total_mse += mse
                num_batches += 1

        average_mse = total_mse / num_batches
        assert average_mse >= 0.0

    def test_visualize_generated_images(self, generator):
        """Test the visualization of generated images.

        This function tests the functionality of the image visualization
        method.

        Args:
            generator (Generator): The generator model.
        """

        try:
            self.visualize_generated_images(generator)
        except Exception as e:
            pytest.fail(f"Error in visualize_generated_images: {e}")


class TestImageProcessor:
    """Test suite for the ImageProcessor class.

    This class tests the functionality of the remove_archipelagos method.

    Methods:
        remove_archipelagos(image, threshold=0.5, min_size=5, min_distance=90):
            Remove small archipelagos from a binary image.
        test_remove_archipelagos():
        Test method for the remove_archipelagos function.

    """

    def remove_archipelagos(self, image, threshold=0.5, min_size=5,
                            min_distance=90):
        """Remove small archipelagos from a binary image.

        Args:
            image (numpy.ndarray): Binary image.
            threshold (float, optional): Threshold value for binarizing
                the image. Defaults to 0.5.
            min_size (int, optional): Minimum size of archipelagos to be
                retained. Defaults to 5.
            min_distance (int, optional): Minimum distance between
                archipelagos. Defaults to 90.

        Returns:
            numpy.ndarray: Cleaned binary image.

        """

        binary_image = image > threshold
        labeled_image, num_features = ndi.label(binary_image)
        sizes = ndi.sum(binary_image, labeled_image, range(num_features + 1))
        mask_size = sizes > min_size
        mask_size[0] = 0
        cleaned_image = mask_size[labeled_image]
        labeled_image, num_features = ndi.label(cleaned_image)
        if num_features >= 2:
            centroids = np.array(ndi.center_of_mass(
                cleaned_image, labeled_image, range(1, num_features + 1)))
            distances = squareform(pdist(centroids))
            upper_triangle_indices = np.triu_indices(num_features, k=1)
            upper_triangle_distances = distances[upper_triangle_indices]
            close_indices = np.where(
                upper_triangle_distances < min_distance)[0]
            for index in close_indices:
                label_to_remove = index + 1
                cleaned_image[labeled_image == label_to_remove] = 0
        return cleaned_image

    def test_remove_archipelagos(self):
        """Test method for the remove_archipelagos function.

        This method tests the functionality of the remove_archipelagos
        function.

        """

        image = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        cleaned_image = self.remove_archipelagos(image)
        assert cleaned_image.sum() == 0, "Test Case 1: Failed"
        image = np.zeros((10, 10))
        cleaned_image = self.remove_archipelagos(image)
        assert np.sum(cleaned_image) == 0, "Test Case 2: Failed"

    def test_remove_archipelagos_threshold(self):
        """Test remove_archipelagos with different threshold values."""
        image = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        cleaned_image = self.remove_archipelagos(image, threshold=0.9)
        assert cleaned_image.sum() == 0, "Test Case 1: Failed"
        cleaned_image = self.remove_archipelagos(image, threshold=0.1)
        assert cleaned_image.sum() == 0, "Test Case 2: Failed"

    def test_remove_archipelagos_min_size(self):
        """Test remove_archipelagos with different minimum size values."""
        image = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        cleaned_image = self.remove_archipelagos(image, min_size=1)
        assert cleaned_image.sum() == 2, "Test Case 1: Failed"
        cleaned_image = self.remove_archipelagos(image, min_size=3)
        assert cleaned_image.sum() == 0, "Test Case 2: Failed"

    def test_remove_archipelagos_min_distance(self):
        """Test remove_archipelagos with different minimum distance values."""
        image = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        cleaned_image = self.remove_archipelagos(image, min_distance=10)
        assert cleaned_image.sum() == 0, "Test Case 1: Failed"
        cleaned_image = self.remove_archipelagos(image, min_distance=100)
        assert cleaned_image.sum() == 0, "Test Case 2: Failed"


class TestImageSelection:
    """Test suite for the ImageSelection class.

    This class tests the functionality of the mse_for_image_selection method.

    Methods:
        mse_for_image_selection(images_to_check, background_image):
            Calculate mean squared error for a list of images relative
            to a background image.
        test_image_selection(): Test method for the mse_for_image_selection
            function.

    """

    def mse_for_image_selection(self, images_to_check, background_image):
        """Calculate mean squared error for a list of images relative
        to a background image.

        Args:
            images_to_check (List[numpy.ndarray]): List of images to compare.
            background_image (numpy.ndarray): Background image.

        Returns:
            List[Tuple[int, float]]: List of tuples containing image index
                and MSE.

        """
        mse_table = []
        for i in range(len(images_to_check)):
            mse = mean_squared_error(
                images_to_check[i].flatten(), background_image.flatten())
            mse_table.append((i, mse))
        mse_table.sort(key=lambda x: x[1])
        return mse_table

    def test_image_selection(self):
        """Test method for the mse_for_image_selection function.

        This method tests the functionality of the mse_for_image_selection
        function.

        """
        images_to_check = [
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])]
        background_image = np.array(
            [[128, 128, 128], [128, 128, 128], [128, 128, 128]])
        mse_table = self.mse_for_image_selection(
            images_to_check, background_image)
        assert len(mse_table) == 2, "Number of Values: Failed"
        assert round(mse_table[0][1], 2) == 16129.0, "Test Case 1: Failed"
        assert round(mse_table[1][1], 2) == 16384.0, "Test Case 2: Failed"

    def test_mse_for_image_selection(self):
        """Test mse_for_image_selection with different input images."""
        images_to_check = [
            np.zeros((3, 3)),
            np.ones((3, 3)) * 255
        ]
        background_image = np.ones((3, 3)) * 128
        mse_table = self.mse_for_image_selection(images_to_check,
                                                 background_image)
        assert len(mse_table) == 2, "Number of Values: Failed"
        assert round(mse_table[0][1], 2) == 16129.0, "Test Case 1: Failed"
        assert round(mse_table[1][1], 2) == 16384.0, "Test Case 2: Failed"

    def test_mse_for_image_selection_empty_input(self):
        """Test mse_for_image_selection with an empty input."""
        mse_table = self.mse_for_image_selection([], np.zeros((3, 3)))
        assert len(mse_table) == 0, "Empty input: Failed"

    def test_mse_for_image_selection_same_image(self):
        """Test mse_for_image_selection with the same image as background."""
        images_to_check = [
            np.zeros((3, 3)),
            np.ones((3, 3)) * 255
        ]
        background_image = np.zeros((3, 3))
        mse_table = self.mse_for_image_selection(images_to_check,
                                                 background_image)
        assert len(mse_table) == 2, "Number of Values: Failed"
        assert round(mse_table[0][1], 2) == 0.0, "Test Case 1: Failed"
        assert round(mse_table[1][1], 2) == 65025.0, "Test Case 2: Failed"


if __name__ == '__main__':
    pytest.main([__file__])
