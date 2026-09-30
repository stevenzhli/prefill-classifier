import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

import torch
import torch.nn.functional as F
from torch import nn, optim

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def _prep_data(y_true, y_probas, classes):
    n_classes = len(classes)

    y_true = pd.Series(y_true)
    
    # Handle list of lists inside a Pandas Series safely
    if isinstance(y_probas, pd.Series) and len(y_probas) > 0 and isinstance(y_probas.iloc[0], list):
        y_probas = y_probas.apply(pd.Series)

    y_probas = np.asarray(y_probas)

    assert y_probas.shape[0] == len(y_true), "Mismatched lengths between y_true and y_probas"
    assert y_probas.shape[1] == n_classes, "y_probas columns do not match number of classes"

    # Safely check for string types in Pandas
    if is_string_dtype(y_true):
        if set(y_true.dropna()).issubset(set(classes)):
            mapper = {v: i for i, v in enumerate(classes)}
            y_true = y_true.map(mapper)
        else:
            raise ValueError("y_true contains string labels not found in the 'classes' list.")
    else:
        y_true = y_true.astype(int)
        if y_true.max() >= n_classes:
            raise ValueError(f"y_true contains an index ({y_true.max()}) larger than allowed by n_classes ({n_classes}).")
    
    # Return y_true as a NumPy array to prevent Pandas index misalignment downstream
    return np.asarray(y_true), y_probas, classes

class MultiClassPlattScaler(BaseEstimator, ClassifierMixin):
    """
    Multi-class Platt Scaling (Matrix Scaling) using Multinomial Logistic Regression.
    Corrects both overall confidence AND class-specific systemic biases.
    """
    def __init__(self, max_iter=1000, C=1.0):
        # Scikit-Learn strictly requires these to be saved as instance variables 
        # with the exact same name as the __init__ arguments!
        self.max_iter = max_iter
        self.C = C
        
        self.model = LogisticRegression(solver='lbfgs', max_iter=self.max_iter, C=self.C)
        self.classes_ = None

    def fit(self, X_logits, y_true):
        """Fits the Platt Scaler to the raw LLM logits."""
        self.model.fit(X_logits, y_true)
        self.classes_ = self.model.classes_
        
        # Optional: Print out the biases to see exactly what the scaler learned!
        print("Learned Class Biases:")
        for cls_idx, bias in zip(self.classes_, self.model.intercept_):
            print(f"Class {cls_idx}: {bias:+.3f}")
            
        return self

    def predict_proba(self, X_logits):
        """Returns perfectly calibrated probabilities."""
        return self.model.predict_proba(X_logits)

    def predict(self, X_logits):
        """Returns the recalibrated ArgMax predictions."""
        return self.model.predict(X_logits)

class TemperatureScaler(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible module to tune temperature to minimize Log Loss.
    Expects X_logits of shape (n_samples, n_classes) and y_true of shape (n_samples,).
    """
    def __init__(self, init_temperature=1.5, max_iter=50, lr=0.01):
        self.init_temperature = init_temperature
        self.max_iter = max_iter
        self.lr = lr
        self.temperature_ = None
        self.classes_ = None

    def fit(self, X_logits, y_true):
        """Fits the optimal temperature scalar to minimize Log Loss."""
        self.classes_ = np.unique(y_true)
        
        logits_tensor = torch.tensor(X_logits, dtype=torch.float32)
        y_tensor = torch.tensor(y_true, dtype=torch.long)
        
        temperature = nn.Parameter(torch.ones(1) * self.init_temperature)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([temperature], lr=self.lr, max_iter=self.max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits_tensor / temperature, y_tensor)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        self.temperature_ = temperature.item()
        
        final_loss = criterion(logits_tensor / self.temperature_, y_tensor).item()
        print(f"Optimal Temperature: {self.temperature_:.4f} | Calibrated Log Loss: {final_loss:.4f}")
        return self

    def predict_proba(self, X_logits):
        """Applies the learned temperature and returns calibrated probabilities."""
        if self.temperature_ is None:
            raise ValueError("TemperatureScaler is not fitted yet.")
            
        logits_tensor = torch.tensor(X_logits, dtype=torch.float32)
        scaled_logits = logits_tensor / self.temperature_
        return F.softmax(scaled_logits, dim=-1).numpy()

    def predict(self, X_logits):
        """Returns ArgMax predictions based on calibrated logits."""
        probas = self.predict_proba(X_logits)
        return np.argmax(probas, axis=-1)

def compute_scaled_probas(logits_list, temperatures):
    """
    Applies temperature scaling and recomputes probabilities (softmax).
    
    :param logits_list: List of numpy arrays (can be 1D arrays per record, or 2D arrays per query)
    :param temperatures: A single float (global temp) OR a list of floats matching logits_list length
    :return: List of probability numpy arrays
    """
    # If a single temperature is passed, apply it to all items in the list
    if isinstance(temperatures, (int, float)):
        temperatures = [temperatures] * len(logits_list)
        
    scaled_probas = []
    
    for logits, temp in zip(logits_list, temperatures):
        # 1. Apply Temperature Scaling
        scaled_logits = logits / temp
        
        # 2. Numerically Stable Softmax (prevents np.exp() overflow)
        # axis=-1 ensures it works perfectly for both 1D and 2D arrays
        max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        
        probas = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        scaled_probas.append(probas)
        
    return scaled_probas

def expected_calibration_error(y_true, y_probas, classes, n_bins=10):
    """
    Calculates the Top-1 Expected Calibration Error (ECE).
    
    ECE is the weighted average of the absolute difference between 
    the model's confidence and its actual accuracy across bins.
    """
    y_true, y_probas, classes = _prep_data(y_true, y_probas, classes)
    
    confidences = np.max(y_probas, axis=1)
    predictions = np.argmax(y_probas, axis=1)
    is_correct = (predictions == y_true).astype(int)
    
    # Create uniform bins between 0 and 1
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Map each confidence score to a bin index
    # (np.clip ensures that a perfect 1.0 probability doesn't overflow the last bin)
    bin_indices = np.clip(np.digitize(confidences, bins) - 1, 0, n_bins - 1)
    
    ece = 0.0
    total_samples = len(confidences)
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        count = mask.sum()
        
        if count > 0:
            bin_accuracy = is_correct[mask].mean()
            bin_confidence = confidences[mask].mean()
            
            # Weighted absolute difference
            ece += (count / total_samples) * np.abs(bin_accuracy - bin_confidence)
            
    return ece