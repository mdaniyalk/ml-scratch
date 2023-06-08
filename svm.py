import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, learning_rate=0.01, max_iters=1000, C=1.0):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.C = C
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize the weight and bias parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Perform stochastic gradient descent to optimize the SVM objective
        for epoch in tqdm(range(self.max_iters)):
            for i in range(n_samples):
                # Choose a random training example
                rand_idx = np.random.randint(0, n_samples)
                X_i, y_i = X[rand_idx], y[rand_idx]
                
                # Calculate the hinge loss and its derivative
                loss = self.C * max(0, 1 - y_i * (np.dot(X_i, self.w) - self.b))
                if loss == 0:
                    grad_w = 0
                    grad_b = 0
                else:
                    grad_w = self.C * y_i * X_i
                    grad_b = -self.C * y_i
                
                # Update the weight and bias parameters using gradient descent
                self.w -= self.learning_rate * (grad_w + 2 * self.w)
                self.b -= self.learning_rate * grad_b
        
    def predict(self, X):
        # Calculate the predicted class labels for the input data
        y_pred = np.sign(np.dot(X, self.w) - self.b)
        return y_pred
