import pytest
import numpy as np
import torch
from torch.nn import MSELoss
from models import ConvLSTMModel
from data import preprocess_data, create_dataloader
from utils import predict_next_background, calculate_mse_1, save_predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_data():
    data = np.random.rand(10, 256, 256)
    return data


@pytest.fixture
def model():
    model = ConvLSTMModel(
        input_channels=1,
        conv_filters=16,
        conv_kernel_size=3,
        hidden_size=128,
        num_layers=2,
        output_size=256*256,
        dropout_rate=0
    )
    model.to(device)
    return model


@pytest.fixture
def test_loader(mock_data):
    processed_data = preprocess_data(mock_data, resample_rate=1)
    dataloader = create_dataloader(processed_data, batch_size=5, shuffle=False)
    return dataloader


@pytest.fixture
def criterion():
    return MSELoss()


def test_preprocess_data(mock_data):
    processed_data = preprocess_data(mock_data, resample_rate=1)
    assert processed_data.shape == (10, 1, 256, 256), \
        "the shape should be (10, 1, 256, 256) after preprocessing"


def test_create_dataloader(mock_data):
    processed_data = preprocess_data(mock_data, resample_rate=1)
    dataloader = create_dataloader(processed_data, batch_size=5, shuffle=False)
    for batch in dataloader:
        assert batch.shape == (5, 1, 256, 256), \
            "the shape of DataLoader is incorrect"
        break


def test_model_output_shape(model, test_loader):
    for inputs in test_loader:
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        assert outputs.shape == (5, 1, 256, 256), \
            "the shape of output is incorrect"
        break


def test_predict_next_background(model, mock_data):
    predictions = predict_next_background(model, mock_data)
    assert len(predictions) == 10, \
        "the number of predicted images and inputs should be consistent"
    assert predictions[0].shape == (256, 256), \
        "the shape of predicted is incorrect"


def test_calculate_mse_1(mock_data):
    predictions = mock_data
    mse_values, average_mse = calculate_mse_1(predictions, mock_data)
    assert average_mse == 0, "the mse should be 0"


def test_save_predictions(tmp_path, mock_data):
    predictions = mock_data
    file_path = tmp_path / "predictions.npy"
    save_predictions(predictions, str(file_path))
    saved_data = np.load(file_path)
    assert np.array_equal(predictions, saved_data), \
        "the data loaded after saving should be the same as the original data"
