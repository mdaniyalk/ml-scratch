import numpy as np
from tqdm import tqdm

class RidgeClassifier:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-3, random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    # @jit
    def softmax(self, scores):
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    # @jit
    def gradient(self, n_samples, X, y_one_hot, proba):
        grad = -1/n_samples * X.T.dot(y_one_hot - proba) + 2 * self.alpha * self.coef_
        return grad

    def fit(self, X, y, verbose=1):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Add column of ones for bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        n_samples, n_features = X.shape
        n_classes = np.unique(y).shape[0]
        
        # Initialize weight matrix
        self.coef_ = np.random.randn(n_features, n_classes)
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1
        
        # Train using gradient descent with L2 regularization
        if verbose == 1:
            iter_range = tqdm(range(self.max_iter))
        elif verbose == 0:
            iter_range = range(self.max_iter)
        for i in iter_range:
            scores = X.dot(self.coef_)
            proba = self.softmax(scores)
            grad = self.gradient(n_samples, X, y_one_hot, proba)
            self.coef_ -= grad * self.tol
            
            # Stop training if change in coefficients is less than tolerance
            if np.linalg.norm(grad) < self.tol:
                break
                
    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        scores = X.dot(self.coef_)
        return np.argmax(scores, axis=1)