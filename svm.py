import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, learning_rate=0.01, max_iters=100, C=0.5):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.C = C
    
    # @jit
    def binary_cross_entropy_loss(self, y_true, y_pred):
        # N = y_true.shape[0]
        N = 1 
        loss = -(1/N) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        return loss

    # @jit
    def sigmoid(self,x):
        return np.exp(-np.logaddexp(0, -x))

    # @jit
    def accuracy(self, y_true, y_pred):
        sum = 0
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                sum = sum + 1
        return sum/len(y_pred)

    # @jit   
    def fit(self, X, y, verbose = 0):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        # Initialize the weight and bias parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Perform stochastic gradient descent to optimize the SVM objective
        for epoch in tqdm(range(self.max_iters), position=0,leave=True, desc="Training SVM Models"):
            for i in range(n_samples):
                # Choose a random training example
                rand_idx = np.random.randint(0, n_samples)
                X_i, y_i = X[rand_idx], y[rand_idx]
                
                # Calculate the binary_cross entropy loss and its derivative
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
            if verbose > 0:
                y_pred = self.sigmoid(np.dot(X, self.w) - self.b)
                acc = self.accuracy(y, y_pred.astype(int))
                print(f'\nEpoch {epoch}/{self.max_iters}: Train Accuracy: {acc}')
        y_pred = self.sigmoid(np.dot(X, self.w) - self.b)
        acc = self.accuracy(y, y_pred.astype(int))
        print(f'\nTrain Accuracy: {acc}')

    def predict(self, X):
        X = np.asarray(X)
        # Calculate the predicted class labels for the input data
        y_pred = self.sigmoid(np.dot(X, self.w) - self.b)
        return y_pred.astype(int)


