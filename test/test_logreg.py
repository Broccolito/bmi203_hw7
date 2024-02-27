"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import matplotlib.pyplot as plt
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from regression import LogisticRegressor

# Helper function to generate synthetic data
def generate_data(n_samples=100, n_features=2):
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features + 1)
    intercept = coef[-1]
    linear_combination = np.dot(X, coef[:-1]) + intercept
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = (probabilities >= 0.5).astype(int)
    return X, y

def test_prediction():
	X, _ = generate_data(10, 2)  # Generating some test data
	model = LogisticRegressor(num_feats=2)
	predictions = model.make_prediction(np.hstack([X, np.ones((X.shape[0], 1))]))  # Adding bias term manually
	assert predictions.shape == (10,), "The prediction should have the same number of elements as the input samples."


def test_loss_function():
    X, y = generate_data(10, 2)
    model = LogisticRegressor(num_feats=2)
    predictions = model.make_prediction(np.hstack([X, np.ones((X.shape[0], 1))]))
    loss = model.loss_function(y, predictions)
    assert isinstance(loss, float), "Loss should be a single floating-point number."


def test_gradient():
    X, y = generate_data(10, 2)
    model = LogisticRegressor(num_feats=2)
    gradient = model.calculate_gradient(y, np.hstack([X, np.ones((X.shape[0], 1))]))
    assert gradient.shape == (3,), "Gradient should have the same shape as the weight vector."


def test_training():
    X, y = generate_data(100, 2)  # Larger dataset for training
    model = LogisticRegressor(num_feats=2, max_iter=10)
    initial_weights = model.W.copy()
    model.train_model(X, y, X, y)  # Using the same dataset for validation for simplicity
    assert not np.array_equal(initial_weights, model.W), "Weights should be updated after training."
