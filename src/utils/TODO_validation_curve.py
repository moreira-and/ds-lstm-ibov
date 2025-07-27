import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.svm import SVC

def validation_curve():
    np.random.seed(0)
    X, y = load_iris(return_X_y=True)
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    train_scores, valid_scores = validation_curve(
        SVC(kernel="linear"), X, y, param_name="C", param_range=np.logspace(-7, 3, 3),
    )
    train_scores
    valid_scores