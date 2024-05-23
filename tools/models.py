import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import random
from livelossplot import PlotLosses
from torchvision.transforms import functional as F
import scipy.ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error

# Global device parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        bool: True if the seed is set successfully.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True


# task 1
def plot_fire_counts(data, title):
    """
    Plot the number of '1's per timestep in the given data.

    Args:
        data (numpy.ndarray):
            The input data array with shape (timesteps, height, width).
        title (str): The title of the plot.

    Returns:
        None
    """
    num_fire_per_timestep = np.sum(data, axis=(1, 2))
    timesteps = np.arange(data.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, num_fire_per_timestep, color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Number of 1s')
    plt.title(title)
    plt.show()


# task 1 test
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            target = inputs[:, -1, :, :, :]
            loss = criterion(outputs, target)
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.9f}")


# task 1
def visualize_predictions(model, dataset, indices):
    """
    Visualize original and reconstructed images using the LSTM model.

    Args:
        model (nn.Module): The LSTM model.
        dataset (Dataset): The dataset containing the images.
        indices (list of int): Indices of the images to visualize.

    Returns:
        None
    """

    model.eval()
    inputs = torch.stack([dataset[i] for i in indices]).to(device)
    inputs = inputs.unsqueeze(1)
    with torch.no_grad():
        outputs = model(inputs)
    outputs = outputs.view(len(indices), 256, 256)
    inputs = inputs[:, -1, 0, :, :]

    plt.figure(figsize=(15, 5))
    for i in range(len(indices)):
        plt.subplot(2, len(indices), i + 1)
        plt.imshow(inputs[i].cpu().numpy(), vmin=0, vmax=1, cmap='copper')
        plt.title(f'Original {indices[i]}')
        plt.axis('off')

        plt.subplot(2, len(indices), i + 1 + len(indices))
        plt.imshow(outputs[i].cpu().numpy(), vmin=0, vmax=1, cmap='copper')
        plt.title(f'Reconstructed {indices[i]}')
        plt.axis('off')
    plt.show()


# task 1
def predict_next_background(model, background_images):
    """
    Predict the next background images using the LSTM model.

    Args:
        model (nn.Module): The LSTM model.
        background_images (numpy.ndarray): Array of background images.

    Returns:
        list of numpy.ndarray: List of predicted background images.
    """
    model.eval()
    predictions = []
    inputs = torch.tensor(
        background_images, dtype=torch.float32).unsqueeze(1).to(device)
    inputs = inputs.unsqueeze(1)

    with torch.no_grad():
        for i in range(len(inputs)):
            input_image = inputs[i]
            predicted_image = model(input_image.unsqueeze(0))
            predicted_image = predicted_image.view(256, 256).cpu().numpy()
            predictions.append(predicted_image)
    return predictions


# task 1 mse
def calculate_mse_1(predicted, actual):
    """
    Calculates MSE for a list of predicted and actual values.

    Parameters:
    predicted : array-like
        List of predicted values.
    actual : array-like
        List of actual values.

    Returns:
    mse_values : list
        List of MSE values for each prediction.
    average_mse : float
        Average MSE value.
    """
    mse_values = [mean_squared_error(pred.flatten(), act.flatten())
                  for pred, act in zip(predicted, actual)]
    average_mse = np.mean(mse_values)
    return mse_values, average_mse


def save_predictions(predictions, file_path):
    """
    Saves predictions to a file in .npy format.

    Parameters:
    predictions : array-like
        Predictions to be saved.
    file_path : str
        Path to save the predictions.
    """
    np.save(file_path, predictions)
    print(f"Predictions saved to {file_path}")


# task 2
def calculate_mse(test_loader, generator_model, device, z_dim=100):
    """
    Calculate the Mean Squared Error between generated images and real images.

    Parameters:
    - test_loader (DataLoader): DataLoader for the test dataset.
    - generator_model (nn.Module): The generator model to create images.
    - device (torch.device): Device to perform calculations on (CPU or GPU).
    - z_dim (int, optional): Dimension of the latent vector. Default is 100.

    Returns:
    - float: The average MSE over the test dataset.
    """
    generator_model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculation
        for images in test_loader:
            images = images.to(device)
            z = torch.randn(images.size(0), z_dim, device=device)
            generated_images = generator_model(z)

            mse = mse_loss(generated_images, images).item()
            total_mse += mse
            num_batches += 1

    average_mse = total_mse / num_batches
    return average_mse


def visualize_generated_images(
        generator_model, device, z_dim=100, num_images_to_generate=5):
    """
    Visualize generated images from the generator model.

    Parameters:
    - generator_model (nn.Module): The generator model to create images.
    - device (torch.device): Device to perform calculations on (CPU or GPU).
    - z_dim (int, optional): Dimension of the latent vector. Default is 100.
    - num_images_to_generate (int, optional):
        Number of images to generate and display. Default is 5.

    Returns:
    - Tensor: Generated images as a tensor.
    """
    generator_model.eval()

    with torch.no_grad():  # Disable gradient calculation
        # Generate latent vectors
        z = torch.randn(num_images_to_generate, z_dim, device=device)
        generated_images = generator_model(z)

    # Display generated images
    _, axes = plt.subplots(1, num_images_to_generate, figsize=(15, 5))
    for idx in range(num_images_to_generate):
        axes[idx].imshow(
            generated_images[idx].cpu().permute(1, 2, 0), cmap="hot")
        axes[idx].set_title("Generated Image")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    return generated_images


def visualize_random_train_images(train_loader, num_images_to_show=5):
    """
    Visualize random images from the training dataset.

    Parameters:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - num_images_to_show (int, optional):
        Number of images to display. Default is 5.

    Returns:
    - None
    """
    # Retrieve a batch of images from the training set
    data_iter = iter(train_loader)
    batch = next(data_iter)

    # Check the structure of the batch
    if isinstance(batch, (list, tuple)) and len(batch) > 1:
        images = batch[0]  # Assuming images are the first element
    else:
        images = batch

    # Select a random subset of images
    indices = random.sample(range(images.size(0)), num_images_to_show)
    selected_images = images[indices]

    # Display the images
    _, axes = plt.subplots(1, num_images_to_show, figsize=(15, 5))
    for idx in range(num_images_to_show):
        axes[idx].imshow(selected_images[idx].permute(1, 2, 0), cmap='hot')
        axes[idx].set_title("Train Image")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


class CenterZoom(object):
    """
    Apply a center zoom transformation to an image.

    Parameters:
    - zoom_factor (float): Factor by which to zoom the image.

    Methods:
    - __call__(img): Apply the center zoom transformation to the input image.

    Example:
    center_zoom = CenterZoom(zoom_factor=2.0)
    transformed_img = center_zoom(img)
    """
    def __init__(self, zoom_factor):
        self.zoom_factor = zoom_factor

    def __call__(self, img):
        # Calculate crop size
        width, height = img.size
        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)
        left = (width - new_width) / 2
        top = (height - new_height) / 2

        # Apply central crop
        img = F.crop(img, top, left, new_height, new_width)
        return img


def remove_archipelagos(image, threshold=0.5, min_size=6, min_distance=90):
    """
    Remove archipelagos (isolated regions) from the image.

    Parameters:
    - image: NumPy array representing the image.
    - threshold: Threshold value to binarize the image.
    - min_size: Minimum size of the region to be considered as a distinct flame
    - min_distance: Minimum distance between regions to consider them distinct.

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
        # Only consider the upper triangle of the distance matrix
        # , excluding the diagonal elements
        upper_triangle_indices = np.triu_indices(num_features, k=1)
        upper_triangle_distances = distances[upper_triangle_indices]

        # Find indices of centroids that are too close
        close_indices = np.where(upper_triangle_distances < min_distance)[0]

        # Remove archipelagos by setting their labels to zero
        for index in close_indices:
            label_to_remove = index + 1  # Labels start from 1
            cleaned_image[labeled_image == label_to_remove] = 0

    return cleaned_image


def mse_for_image_selection(images_to_check, background_image):
    """
    Calculate the MSE between a set of images and a background image,
    and return the results sorted by MSE values.

    Parameters:
    - images_to_check (list of ndarray):
        List of images to compare against the background image.
    - background_image (ndarray): The background image to compare against.

    Returns:
    - list of tuples: Sorted list of tuples where each tuple contains:
        - int: Index of the image in `images_to_check`.
        - float: Corresponding MSE value.
    """
    mse_values = []
    mse_table = []

    for i in range(len(images_to_check)):
        mse = mean_squared_error(
            images_to_check[i].flatten(), background_image.flatten())
        mse_values.append(mse)

        # Store these values in a table associated with the image number
        mse_table.append((i, mse))

    # Sort the table by the MSE values
    mse_table.sort(key=lambda x: x[1])

    return mse_table


# Global device parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# task 1 model
class ConvLSTMModel(nn.Module):
    """
    Convolutional LSTM model for sequence prediction.
    """
    def __init__(self, input_channels, conv_filters, conv_kernel_size,
                 hidden_size, num_layers, output_size, dropout_rate=0.5):
        """
        Initialize ConvLSTMModel.

        Parameters:
        input_channels : int
            Number of input channels.
        conv_filters : int
            Number of convolutional filters.
        conv_kernel_size : int
            Size of convolutional kernel.
        hidden_size : int
            Size of the hidden state of LSTM.
        num_layers : int
            Number of LSTM layers.
        output_size : int
            Size of the output.
        dropout_rate : float, optional
            Dropout rate, by default 0.5.
        """

        super(ConvLSTMModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, conv_filters,
                      kernel_size=conv_kernel_size,
                      padding=conv_kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv_filters, conv_filters,
                      kernel_size=conv_kernel_size,
                      padding=conv_kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        conv_output_size = (256 // 4) * (256 // 4) * conv_filters
        self.lstm = nn.LSTM(conv_output_size, hidden_size,
                            num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        Forward pass of ConvLSTMModel.

        Parameters:
        input : torch.Tensor
            Input tensor.

        Returns:
        torch.Tensor
            Output tensor.
        """
        batch_size, seq_length, channels, height, width = input.size()
        c_in = input.view(batch_size * seq_length, channels, height, width)
        c_out = self.conv(c_in)
        c_out = c_out.view(batch_size, seq_length, -1)
        h0 = torch.zeros(self.lstm.num_layers, batch_size,
                         self.lstm.hidden_size).to(input.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size,
                         self.lstm.hidden_size).to(input.device)
        out, _ = self.lstm(c_out, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.view(batch_size, 1, 256, 256)


def train_and_validate(model, train_loader, test_loader,
                       criterion, optimizer, num_epochs, accumulation_steps=2):
    """
    Train and validate the model.

    Parameters:
    model : nn.Module
        The model to train and validate.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    test_loader : torch.utils.data.DataLoader
        DataLoader for testing data.
    criterion : function
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    num_epochs : int
        Number of epochs.
    accumulation_steps : int, optional
        Accumulation steps for gradient accumulation, by default 2.
    """

    liveloss = PlotLosses()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Clear gradients at the start of each epoch
        for i, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            target = inputs.view(inputs.size(0), 1, 256, 256)
            loss = criterion(outputs, target)
            torch.cuda.empty_cache()
            loss.backward()
            torch.cuda.empty_cache()
            if (i + 1) % accumulation_steps == 0:
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                target = inputs.view(inputs.size(0), 1, 256, 256)
                loss = criterion(outputs, target)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)

        logs = {'log loss': train_loss, 'val_log loss': val_loss}
        liveloss.update(logs)
        liveloss.draw()

        print(f"Epoch {epoch+1}/{num_epochs},\
              Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'lstm_model.pth')
    print("Model saved to lstm_model.pth")


# task 2 model
class Generator_basic_GAN(nn.Module):
    """
    Basic Generator model for a GAN.

    Parameters:
    - g_input_dim (int, optional): Dimension of the input noise vector.
    - g_output_dim (int, optional): Dimension of the output image.
    """
    def __init__(self, g_input_dim=100, g_output_dim=256*256):
        super().__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        """
        Forward pass through the generator.

        Parameters:
        - x (Tensor): Input noise vector.

        Returns:
        - Tensor: Generated image.
        """
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator_basic_GAN(nn.Module):
    """
    Basic Discriminator model for a GAN.

    Parameters:
    - d_input_dim (int, optional): Dimension of the input image.
    """
    def __init__(self, d_input_dim=256*256):
        super().__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Parameters:
        - x (Tensor): Input image.

        Returns:
        - Tensor: Discriminator output (probability that the image is real).
        """
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def build_model(Generator, Discriminator, z_dim=100, lr=0.0001):
    """
    Build and initialize the Generator and Discriminator
    models along with their optimizers.

    Parameters:
    - Generator (nn.Module): The Generator class.
    - Discriminator (nn.Module): The Discriminator class.
    - z_dim (int, optional): Dimension of the input noise vector.
    - lr (float, optional): Learning rate for the optimizers.

    Returns:
    - tuple: (Generator model, Discriminator model,
              Generator optimizer, Discriminator optimizer)
    """
    G = Generator().to(device)
    D = Discriminator().to(device)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)

    return G, D, G_optimizer, D_optimizer


def D_train_basic(G, D, D_optimizer, input_data, device,
                  criterion=nn.BCELoss(), batch_size=100, z_dim=100):
    """
    Train the Discriminator with real and fake data for one step.

    Parameters:
    - D (nn.Module): Discriminator model.
    - D_optimizer (Optimizer): Optimizer for the Discriminator.
    - input_data (Tensor): Real input data.
    - device (torch.device): Device to run the calculations on.
    - criterion (nn.Module, optional): Loss function. Default is nn.BCELoss().
    - batch_size (int, optional): Batch size. Default is 100.
    - z_dim (int, optional): Dimension of the input noise vector.

    Returns:
    - float: Loss value for the Discriminator.
    """
    D.train()
    D_optimizer.zero_grad()

    # Train discriminator on real data
    x_real, y_real = input_data.view(-1, 256*256), torch.ones(batch_size, 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # Train discriminator on fake data
    z = torch.randn(batch_size, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # Combine the losses
    D_loss = D_real_loss + D_fake_loss

    # Model update
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train_basic(G, D, G_optimizer, batch_size=100,
                  z_dim=100, criterion=nn.BCELoss()):
    """
    Train the Generator for one step.

    Parameters:
    - G (nn.Module): Generator model.
    - G_optimizer (Optimizer): Optimizer for the Generator.
    - batch_size (int, optional): Batch size. Default is 100.
    - z_dim (int, optional): Dimension of the input noise vector.
    - criterion (nn.Module, optional): Loss function. Default is nn.BCELoss().

    Returns:
    - float: Loss value for the Generator.
    """
    G.train()
    G_optimizer.zero_grad()

    # Sample vector and produce generator output
    z = torch.randn(batch_size, z_dim).to(device)
    G_output = G(z)

    # Obtain scores from D for the generated data
    D_output = D(G_output)

    # Train generator to "fool" discriminator
    y = torch.ones(batch_size, 1).to(device)
    G_loss = criterion(D_output, y)

    # Model update
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


class Generator_DCGAN_first_attempt(nn.Module):
    """
    Generator model for a DCGAN.

    Parameters:
    - g_input_dim (int, optional): Dimension of the input noise vector
    - g_output_dim (int, optional): Dimension of the output image
    """
    def __init__(self, g_input_dim=100, g_output_dim=256):
        super().__init__()
        self.init_size = g_output_dim // 16
        self.fc1 = nn.Linear(g_input_dim, 128 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass through the generator.

        Parameters:
        - x (Tensor): Input noise vector.

        Returns:
        - Tensor: Generated image.
        """
        x = self.fc1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator_DCGAN_first_attempt(nn.Module):
    """
    Discriminator model for a DCGAN.

    Parameters:
    - d_input_dim (int, optional): Dimension of the input image.
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
        """
        Forward pass through the discriminator.

        Parameters:
        - x (Tensor): Input image.

        Returns:
        - Tensor: Discriminator output (probability that the image is real).
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.adv_layer(x)
        return x


def D_train(input_data, D, G, D_optimizer, device,
            batch_size=100, criterion=nn.BCELoss(), z_dim=100):
    """
    Train the Discriminator with real and fake data for one step.

    Parameters:
    - input_data (Tensor): Real input data.
    - D (nn.Module): Discriminator model.
    - G (nn.Module): Generator model.
    - D_optimizer (Optimizer): Optimizer for the Discriminator.
    - device (torch.device): Device to run the calculations on.
    - batch_size (int, optional): Batch size. Default is 100.
    - criterion (nn.Module, optional): Loss function. Default is nn.BCELoss().
    - z_dim (int, optional): Dimension of the input noise vector

    Returns:
    - float: Loss value for the Discriminator.
    """
    D.train()
    D_optimizer.zero_grad()

    # Train discriminator on real data
    x_real, y_real = input_data.view(
        -1, 1, 256, 256), torch.ones(batch_size, 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # Train discriminator on fake data
    z = torch.randn(batch_size, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # Combine the losses
    D_loss = D_real_loss + D_fake_loss

    # Model update
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(D, G, G_optimizer, device, batch_size=100,
            criterion=nn.BCELoss(), z_dim=100):
    """
    Train the Generator for one step.

    Parameters:
    - G (nn.Module): Generator model.
    - G_optimizer (Optimizer): Optimizer for the Generator.
    - device (torch.device): Device to run the calculations on.
    - batch_size (int, optional): Batch size. Default is 100.
    - criterion (nn.Module, optional): Loss function. Default is nn.BCELoss().
    - z_dim (int, optional): Dimension of the input noise vector.

    Returns:
    - float: Loss value for the Generator.
    """
    G.train()
    G_optimizer.zero_grad()

    # Sample vector and produce generator output
    z = torch.randn(batch_size, z_dim).to(device)
    G_output = G(z)

    # Obtain scores from D for the generated data
    D_output = D(G_output)

    # Train generator to "fool" discriminator
    _ = torch.ones(batch_size, 1).to(device)
    G_loss = criterion(D_output, torch.ones_like(D_output))

    # Model update
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def model_runner(D, G, D_optimizer, G_optimizer, epochs,
                 epochs_to_save, train_loader, model_name, device):
    """
    Run the training process for the GAN models.

    Parameters:
    - epochs (int): Number of training epochs.
    - epochs_to_save (int): Frequency of saving the model (every n epochs).
    - train_loader (DataLoader): DataLoader for the training dataset.
    - model_name (str): Name for saving the model files.
    - device (torch.device): Device to run the calculations on.
    """
    n_epoch = epochs
    groups = {'Loss': ['D_Loss', 'G_Loss']}
    liveloss = PlotLosses(groups=groups)

    for epoch in range(1, n_epoch + 1):
        D_losses, G_losses = [], []
        logs = {}

        for _, real_images in enumerate(train_loader):
            real_images = real_images.view(-1, 256, 256).to(device)
            D_loss = D_train(real_images, D, G, D_optimizer, device)
            G_loss = G_train(D, G, G_optimizer, device)

            D_losses.append(D_loss)
            G_losses.append(G_loss)

        logs['D_Loss'] = torch.tensor(D_losses).mean().item()
        logs['G_Loss'] = torch.tensor(G_losses).mean().item()

        liveloss.update(logs)
        liveloss.draw()

        if epoch % epochs_to_save == 0:
            torch.save(
                G.state_dict(), f"/content/gdrive/MyDrive/\
                    {model_name}_{epoch:03d}.pth")


class Generator_final_model(nn.Module):
    """
    Final Generator model for the GAN.

    Parameters:
    - g_input_dim (int, optional): Dimension of the input noise vector.
    - g_output_dim (int, optional): Dimension of the output image.
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
        """
        Forward pass through the generator.

        Parameters:
        - x (Tensor): Input noise vector.

        Returns:
        - Tensor: Generated image.
        """
        x = self.fc1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator_final_model(nn.Module):
    """
    Final Discriminator model for the GAN.

    Parameters:
    - d_input_dim (int, optional): Dimension of the input image.
    """
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
        """
        Forward pass through the discriminator.

        Parameters:
        - x (Tensor): Input image.

        Returns:
        - Tensor: Discriminator output (probability that the image is real).
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.adv_layer(x)
        return x


class GaussianNoise(nn.Module):
    """
    Apply Gaussian noise to an input tensor.

    Parameters:
    - sigma (float, optional): Standard deviation of the Gaussian noise.
    """
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        """
        Forward pass through the Gaussian noise layer.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Tensor with added Gaussian noise.
        """
        if self.training:
            noise = x.new_empty(x.size()).normal_(mean=0, std=self.sigma)
            return x + noise
        return x


class Generator_noisy(nn.Module):
    """
    Generator model for a GAN with Gaussian noise.

    Parameters:
    - g_input_dim (int, optional): Dimension of the input noise vector.
    - g_output_dim (int, optional): Dimension of the output image.
    """
    def __init__(self, g_input_dim=100, g_output_dim=256):
        super().__init__()
        self.init_size = g_output_dim // 16
        self.fc1 = nn.Linear(g_input_dim, 128 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            GaussianNoise(sigma=0.1),
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
        """
        Forward pass through the generator.

        Parameters:
        - x (Tensor): Input noise vector.

        Returns:
        - Tensor: Generated image.
        """
        x = self.fc1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x
