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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from regression import LogisticRegressor


# Load data
print('Loading data into the workspace...')
X_train, X_val, y_train, y_val = utils.loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ],
    split_percent=0.8,
    split_seed=42
)

# Scale the data, since values vary across feature. Note that we
# fit on the training data and use the same scaler for X_val.
print('Scaling the data...')
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

# For testing purposes, once you've added your code.
# CAUTION: hyperparameters have not been optimized.
print('Showing loss curve...')
log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=10000, batch_size=10)
log_model.train_model(X_train, y_train, X_val, y_val)

y_pred_val = log_model.make_prediction(np.hstack([X_val, np.ones((X_val.shape[0], 1))]))
print("Prediction values for validation data:", y_pred_val)

# Print Loss Values
# Last training loss
last_train_loss = log_model.loss_hist_train[-1]
print("Last training loss:", last_train_loss)
# Last validation loss
last_val_loss = log_model.loss_hist_val[-1]
print("Last validation loss:", last_val_loss)