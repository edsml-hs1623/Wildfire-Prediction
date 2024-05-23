import numpy as np
from numpy.linalg import inv


def update_prediction(x, K, H, y):
    """
    Update the state estimate in a Kalman Filter.

    Parameters:
    x (numpy.ndarray): The prior state estimate.
    K (numpy.ndarray): The Kalman Gain matrix.
    H (numpy.ndarray): The observation matrix.
    y (numpy.ndarray): The observed measurement.

    Returns:
    numpy.ndarray: The updated state estimate.
    """
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def KalmanGain(B, H, R):
    """
    Calculate the Kalman Gain.

    Parameters:
    B (numpy.ndarray): The estimate error covariance matrix.
    H (numpy.ndarray): The observation matrix.
    R (numpy.ndarray): The measurement noise covariance matrix.

    Returns:
    numpy.ndarray: The Kalman Gain matrix.
    """
    tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
    res = np.dot(B, np.dot(H.transpose(), tempInv))
    return res


def mse(y_obs, y_pred):
    """
    Calculate the Mean Squared Error (MSE)
      between observed and predicted values.

    Parameters:
    y_obs (numpy.ndarray): The observed values.
    y_pred (numpy.ndarray): The predicted values.

    Returns:
    float: The Mean Squared Error.
    """
    return np.square(np.subtract(y_obs, y_pred)).mean()


def test_update_prediction():
    x = np.array([1.0, 2.0])
    K = np.array([[0.5, 0.0], [0.0, 0.5]])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([2.0, 3.0])
    expected_result = np.array([1.5, 2.5])
    result = update_prediction(x, K, H, y)
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_KalmanGain():
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = np.array([[0.1, 0.0], [0.0, 0.1]])
    expected_result = np.array([[0.90909091, 0.0], [0.0, 0.90909091]])
    result = KalmanGain(B, H, R)
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_mse():
    y_obs = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    expected_result = 0.0
    result = mse(y_obs, y_pred)
    assert result == expected_result

    y_obs = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    expected_result = 0.02  # Corrected expected result
    result = mse(y_obs, y_pred)
    assert np.isclose(result, expected_result, atol=1e-5)


if __name__ == "__main__":
    import pytest
    pytest.main()
