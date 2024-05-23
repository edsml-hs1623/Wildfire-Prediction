# utils.py
import numpy as np
import time
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.transforms import functional as F
import scipy.ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv

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


def calculate_mse_1(predicted, actual):
    mse_values = [mean_squared_error(
        pred.flatten(), act.flatten()) for pred, act in zip(predicted, actual)]
    average_mse = np.mean(mse_values)
    return mse_values, average_mse


def save_predictions(predictions, file_path):
    np.save(file_path, predictions)
    print(f"Predictions saved to {file_path}")


def calculate_mse(test_loader, generator_model, device, z_dim=100):
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


def calculate_mse(test_loader, generator_model, device, z_dim=100):
    """
    Calculate the Mean Squared Error (MSE)
     between generated images and real images.

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
    fig, axes = plt.subplots(1, num_images_to_generate, figsize=(15, 5))
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
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(15, 5))
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
        # right = (width + new_width) / 2
        # bottom = (height + new_height) / 2

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
        centroids = np.array(
            ndi.center_of_mass(cleaned_image, labeled_image, range(
                1, num_features + 1)))

        # Compute pairwise distances between centroids
        distances = squareform(pdist(centroids))

        # Check if any distances are smaller than the minimum distance
        # Only consider the upper triangle of the distance matrix,
        #  excluding the diagonal
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
    Calculate the Mean Squared Error (MSE) between a set of
    images and a background image,
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


def update_prediction(x, K, H, y):
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def KalmanGain(B, H, R):
    tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
    res = np.dot(B, np.dot(H.transpose(), tempInv))
    return res


def mse(y_obs, y_pred):
    return np.square(np.subtract(y_obs, y_pred)).mean()


def data_assimilation(
        model_compressed,
        sensor_compressed,
        sensor,
        background,
        sensor_1D,
        pca,
        B,
        H,
        R):
    start = time.time()
    # Excusion Time in Reduced Space
    K = KalmanGain(B, H, R)
    updated_data = np.array([update_prediction(md, K, H, sd) for md, sd in zip(
        model_compressed, sensor_compressed)])
    end1 = time.time()
    # Excusion Time in Physical Space
    updated_recoverd = pca.inverse_transform(updated_data)
    # updated_recoverd_model = sensor_1.inverse_transform(updated_data)
    end2 = time.time()
    mse_before_DA_redu = mse(sensor_compressed, model_compressed)
    mse_after_DA_redu = mse(sensor_compressed, updated_data)
    print('MSE before DA in Reduced Space:', mse_before_DA_redu)
    print('MSE after DA in Reduced Space:', mse_after_DA_redu)
    print(f'Time for DA: {end1 - start} seconds')
    print('----------------------------------------')
    mse_before_DA_physical = mse(sensor, background)
    mse_after_DA_physical = mse(sensor_1D, updated_recoverd)
    print('MSE before DA in Physical Space:', mse_before_DA_physical)
    print('MSE after DA in Physical Space:', mse_after_DA_physical)
    print(f'Time for DA + Decompression: {end2 - start} seconds')

    return updated_recoverd


def plot_data(sensor, background, updated_recovered):
    # Reshape the recovered data
    updated_recovered_2D = np.reshape(
        updated_recovered, (sensor.shape[0], sensor.shape[1], sensor.shape[2]))
    error = sensor - updated_recovered_2D

    # Set up the figure and axes
    fig, axes = plt.subplots(4, 4, figsize=[20, 20])

    # Define the data sets and titles
    data_sets = [sensor, background, updated_recovered_2D, error]
    titles = ["Sensor", "Predicted - background", "Assimilated data", "Error"]

    # Loop through the rows and columns to create subplots
    for row, (data_set, title) in enumerate(zip(data_sets, titles)):
        for col in range(4):
            ax = axes[row, col]
            im = ax.imshow(data_set[col], vmin=0.0, vmax=1.0, cmap='hot')
            ax.set_title(f"{title} {col}")
            fig.colorbar(im, ax=ax)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def scale_and_plot_B(
        base_B, scaling_factors, model_encoded, sensor_encoded, H, R):
    """
    Perform data assimilation (DA) with different scaling
    factors for the covariance matrix B
    and plot the results.

    Args:
    - scaling_factors: numpy array or list of scaling factors for B
    - model_encoded: Encoded model data
    - sensor_encoded: Encoded sensor data
    - H: Transformation matrix
    - R: Covariance matrix of sensor data

    Returns:
    - None
    """
    # Lists to store MSE results
    mse_before_DA_list_cov = []
    mse_after_DA_list_cov = []

    print("Start Assimilation")

    for scale in scaling_factors:
        # Method 1: Using covariance matrix scaled by `scale`
        B_cov = base_B * scale

        K_cov = KalmanGain(B_cov, H, R)

        updated_data_list_cov = [
            update_prediction(model_encoded[i], K_cov, H, sensor_encoded[i])
            for i in range(len(model_encoded))
        ]

        updated_data_array_cov = np.array(updated_data_list_cov)

        mse_before_DA_cov = mse(model_encoded, sensor_encoded)
        mse_after_DA_cov = mse(updated_data_array_cov, sensor_encoded)

        mse_before_DA_list_cov.append(mse_before_DA_cov)
        mse_after_DA_list_cov.append(mse_after_DA_cov)

        # Print scale factor with associated MSE
        print(
            f"Scale factor (cov): {scale:.2f},\
                  MSE before DA: {mse_before_DA_cov:.4f},\
                      MSE after DA: {mse_after_DA_cov:.4f}")

    print("Assimilation completed")

    # Plotting the results
    plt.figure(figsize=(8, 6))

    # Plotting results for covariance matrix scaling
    plt.plot(scaling_factors,
             mse_after_DA_list_cov, label='MSE after DA', marker='o')
    plt.xlabel('Scaling factor for covariance matrix B')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE for Covariance Matrix Scaling')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# helper function to select R
def scale_and_plot_R(base_R,
                     scaling_factors,
                     model_encoded, sensor_encoded, H, B):
    """
    Perform data assimilation (DA) with different scaling factors for
      the covariance matrix R
    and plot the results.

    Args:
    - scaling_factors: numpy array or list of scaling factors for B
    - model_encoded: Encoded model data
    - sensor_encoded: Encoded sensor data
    - H: Transformation matrix
    - B: Covariance matrix of model data

    Returns:
    - None
    """
    # Lists to store MSE results
    mse_before_DA_list_cov = []
    mse_after_DA_list_cov = []

    print("Start Assimilation")

    for scale in scaling_factors:
        # Method 1: Using covariance matrix scaled by `scale`
        R_cov = base_R * scale

        K_cov = KalmanGain(B, H, R_cov)

        updated_data_list_cov = [
            update_prediction(model_encoded[i], K_cov, H, sensor_encoded[i])
            for i in range(len(model_encoded))
        ]

        updated_data_array_cov = np.array(updated_data_list_cov)

        mse_before_DA_cov = mse(model_encoded, sensor_encoded)
        mse_after_DA_cov = mse(updated_data_array_cov, sensor_encoded)

        mse_before_DA_list_cov.append(mse_before_DA_cov)
        mse_after_DA_list_cov.append(mse_after_DA_cov)

        # Print scale factor with associated MSE
        print(f"Scale factor (cov): {scale:.2f},\
               MSE before DA: {mse_before_DA_cov:.4f},\
                  MSE after DA: {mse_after_DA_cov:.4f}")

    print("Assimilation completed")

    # Plotting the results
    plt.figure(figsize=(8, 6))

    # Plotting results for covariance matrix scaling
    plt.plot(scaling_factors, mse_after_DA_list_cov,
             label='MSE after DA', marker='o')
    plt.xlabel('Scaling factor for covariance matrix R')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE for Covariance Matrix Scaling')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
