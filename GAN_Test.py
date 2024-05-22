import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error


# TESTING DATALOADER
class NumpyDataset(Dataset):
    """Dataset class for loading numpy data."""
    def __init__(self, data_path, transform=None):
        self.data = np.load(data_path).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image


class TestDataLoader(unittest.TestCase):
    """Test suite for the DataLoader."""
    def setUp(self):
        self.batch_size = 100

        # Create dummy data for testing
        self.dummy_train = np.random.randn(200, 256, 256).astype(np.float32)
        self.dummy_test = np.random.randn(100, 256, 256).astype(np.float32)
        np.save('dummy_train.npy', self.dummy_train)
        np.save('dummy_test.npy', self.dummy_test)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.train_data = NumpyDataset('dummy_train.npy', self.transform)
        self.test_data = NumpyDataset('dummy_test.npy', self.transform)

        self.train_loader = DataLoader(self.train_data, self.batch_size, True)
        self.test_loader = DataLoader(self.test_data, self.batch_size, True)

    def test_train_loader(self):
        print("\n----------Testing Train Loader----------")
        for i, batch in enumerate(self.train_loader):
            self.assertEqual(
                batch.shape[0],  self.batch_size,
                msg=f"Batch {i+1}: Train loader batch size mismatch")
            print("Batch Size: Passed")

            self.assertEqual(
                batch.shape[1:], (1, 256, 256),
                msg=f"Batch {i+1}: Train loader batch shape mismatch")
            print("Batch Shape: Passed")

            self.assertIsInstance(
                batch, torch.Tensor,
                msg=f"Batch {i+1}: Train loader batch is not a Tensor")
            print("Tensor Instance: Passed")
            self.assertEqual(
                batch.dtype, torch.float32,
                msg=f"Batch {i+1}: Train loader batch dtype mismatch")
            print("Data Type: Passed")

            if i >= 0:  # Adjust this number to test more batches
                break  # Test only the first batch to keep the test fast

    def test_test_loader(self):
        print("\n----------Testing Test Loader----------")
        for i, batch in enumerate(self.test_loader):
            self.assertEqual(
                batch.shape[0], self.batch_size,
                msg=f"Batch {i+1}: Test loader batch size mismatch")
            print("Batch Size: Passed")

            self.assertEqual(
                batch.shape[1:], (1, 256, 256),
                msg=f"Batch {i+1}: Test loader batch shape mismatch")
            print("Batch Shape: Passed")

            self.assertIsInstance(
                batch, torch.Tensor,
                msg=f"Batch {i+1}: Test loader batch is not a Tensor")
            print("Tensor Instance: Passed")

            self.assertEqual(
                batch.dtype, torch.float32,
                msg=f"Batch {i+1}: Test loader batch dtype mismatch")
            print("Data Type: Passed")
            if i >= 0:  # Adjust this number to test more batches
                break  # Test only the first batch to keep the test fast


# TESTING GAN
class Generator(nn.Module):
    """Generator model for GAN."""
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
        x = self.fc1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    """Discriminator model for GAN."""
    def __init__(self, d_input_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 16, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            spectral_norm(nn.Conv2d(16, 32, 3, 2, 1)),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            spectral_norm(nn.Conv2d(32, 64, 3, 2, 1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            spectral_norm(nn.Conv2d(64, 128, 3, 1, 1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        ds_size = d_input_dim // 8
        self.linear_input_size = 128 * ds_size ** 2
        self.adv_layer = nn.Sequential(
            nn.Linear(self.linear_input_size, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.adv_layer(x)
        return x


class TestGAN(unittest.TestCase):
    """Test suite for the GAN."""
    def setUp(self):
        self.batch_size = 16
        self.g_input_dim = 100
        self.d_input_dim = 256

        self.generator = Generator(g_input_dim=self.g_input_dim,
                                   g_output_dim=self.d_input_dim)
        self.discriminator = Discriminator(d_input_dim=self.d_input_dim)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.test_data = NumpyDataset('dummy_test.npy', self.transform)
        self.test_loader = DataLoader(self.test_data, self.batch_size, True)

    @staticmethod
    def visualize_generated_images(generator_model, 
                                   z_dim=100, num_images_to_generate=5):
        """Generation of Images"""

        generator_model.eval()

        with torch.no_grad():  # Disable gradient calculation
            # Generate latent vectors
            z = torch.randn(num_images_to_generate, z_dim)
            generated_images = generator_model(z)

        return generated_images

    def test_generator(self):
        print("\n-----------Testing Generator------------")
        noise = torch.randn(self.batch_size, self.g_input_dim)
        generated_data = self.generator(noise)
        self.assertEqual(
            generated_data.shape,
            (self.batch_size, 1, self.d_input_dim, self.d_input_dim),
            msg=(f"Expected({self.batch_size}, 1, {self.d_input_dim}, "
                 f"{self.d_input_dim}), got {generated_data.shape}")
        )
        print("Output Shape: Passed")

        try:
            self.generator(noise)
            print("Functionality: Passed")
        except Exception as e:
            self.fail(f"Generator failed to run: {e}")

    def test_discriminator(self):
        print("\n----------Testing Discriminator----------")
        images = torch.randn(
            self.batch_size, 1, self.d_input_dim, self.d_input_dim)
        output = self.discriminator(images)
        self.assertEqual(
            output.shape, (self.batch_size, 1),
            msg=f"Expected ({self.batch_size}, 1), got {output.shape}")
        print("Output Shape: Passed")

        try:
            self.discriminator(images)
            print("Functionality: Passed")
        except Exception as e:
            self.fail(f"Discriminator failed to run: {e}")

    def test_mse_calculation(self):
        print("\n----------Testing MSE Calculation----------")
        self.generator.eval()  # Set the generator to evaluation mode
        mse_loss = nn.MSELoss()
        total_mse = 0.0
        num_batches = 0

        with torch.no_grad():
            for images in self.test_loader:
                images = images
                noise = torch.randn(images.size(0), self.g_input_dim)
                generated_images = self.generator(noise)

                mse = mse_loss(generated_images, images).item()
                total_mse += mse
                num_batches += 1

        average_mse = total_mse / num_batches
        self.assertGreaterEqual(
            average_mse, 0.0,
            msg="Mean Squared Error should be non-negative.")
        if average_mse > 0.0:
            print("Non-Negative: Passed")

    def test_visualize_generated_images(self):
        print("\n------------Testing Visualizing Generated Images------------")
        try:
            self.visualize_generated_images(self.generator)
            print("Functionality: Passed")
        except Exception as e:
            self.fail(f"Error in visualize_generated_images: {e}")


# TESTING IMAGES CLEANING
class TestImageProcessor(unittest.TestCase):
    """Class for image processing."""

    def remove_archipelagos(self, image,
                            threshold=0.5, min_size=5, min_distance=90):
        """
        Remove archipelagos (isolated regions) from the image.

        Parameters:
        - image: NumPy array representing the image
        - threshold: Threshold value to binarize the image
        - min_size: Min size of the region to be considered as a distinct flame
        - min_distance: Min distance between regions to consider them distinct

        Returns:
        - The cleaned image with archipelagos removed.
        """
        # Step 1: Convert to binary image based on the threshold
        binary_image = image > threshold

        # Step 2: Label connected components
        labeled_image, num_features = ndi.label(binary_image)

        # Step 3: Filter out small regions
        sizes = ndi.sum(binary_image, labeled_image, range(num_features + 1))
        mask_size = sizes > min_size
        mask_size[0] = 0  # Remove background label
        cleaned_image = mask_size[labeled_image]

        # Step 4: Recount connected components in the cleaned image
        labeled_image, num_features = ndi.label(cleaned_image)

        # Step 5: Check if there are two or more distinct regions
        if num_features >= 2:
            # Compute the centroids of the labeled regions
            centroids = np.array(ndi.center_of_mass(
                cleaned_image, labeled_image, range(1, num_features + 1)))

            # Compute pairwise distances between centroids
            distances = squareform(pdist(centroids))

            # Check if any distances are smaller than the minimum distance
            # Only consider the upper triangle of the distance matrix,
            # excluding the diagonal one
            upper_triangle_indices = np.triu_indices(num_features, k=1)
            upper_triangle_distances = distances[upper_triangle_indices]

            # Find indices of centroids that are too close
            close_indices = np.where(
                upper_triangle_distances < min_distance)[0]

            # Remove archipelagos by setting their labels to zero
            for index in close_indices:
                label_to_remove = index + 1  # Labels start from 1
                cleaned_image[labeled_image == label_to_remove] = 0

        return cleaned_image

    def test_remove_archipelagos(self):
        print("\n----------- Testing Remove Archipelagos ------------")
        """Test Case 1: Random Sample Image"""

        # Create a sample image with archipelagos
        image = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]])
        
        # Remove archipelagos from the image
        cleaned_image = self.remove_archipelagos(image)
        
        # Ensure that the cleaned image does not contain any archipelagos
        self.assertEqual(
            cleaned_image.sum(), 8,
            msg="Cleaned image should contain 8 non-zero elements.")
        print("Test Case 1: Passed")

        """Test Case 2: Empty Sample Image"""
        image = np.zeros((10, 10))

        # Instantiate the ImageProcessor class
        image_processor = TestImageProcessor()

        # Remove archipelagos from the empty image
        cleaned_image = image_processor.remove_archipelagos(image)

        # Ensure that the cleaned image is also empty
        self.assertEqual(
            np.sum(cleaned_image), 0,
            msg="Cleaned image should be empty.")

        print("Test Case 2: Passed")


# TESTING IMAGES SELECTION
class TestImageSelection(unittest.TestCase):
    """Test suite for the mse_calc function."""

    def image_selection(self, images_to_check, background_image):
        mse_values = []
        mse_table = []

        for i in range(len(images_to_check)):
            mse = mean_squared_error(
                images_to_check[i].flatten(), background_image.flatten())
            mse_values.append(mse)

            # store these values in a table associated with the image number
            mse_table.append((i, mse))

        # sort the table by the MSE values
        mse_table.sort(key=lambda x: x[1])

        return mse_table

    def test_image_selection(self):
        print("\n-----------Testing Image Selection------------")
        # Create some sample images to check
        images_to_check = [
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])]

        # Create a sample background image
        background_image = np.array(
            [[128, 128, 128], [128, 128, 128], [128, 128, 128]])

        # Calculate MSE values
        mse_table = self.image_selection(images_to_check, background_image)

        # Check if the MSE values are calculated correctly
        self.assertEqual(
            len(mse_table), 2,
            msg="Incorrect number of MSE values calculated.")
        print("Number of Values: Passed")

        self.assertAlmostEqual(
            mse_table[0][1], 16129.0, places=2,
            msg="Incorrect MSE value for the first image.")
        print("Test Case 1: Passed")

        self.assertAlmostEqual(
            mse_table[1][1], 16384.0, places=2,
            msg="Incorrect MSE value for the second image.")
        print("Test Case 2: Passed")


if __name__ == '__main__':
    unittest.main()
