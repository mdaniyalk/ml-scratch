import numpy as np

class RidgeCV:
    def __init__(self, alphas=np.logspace(-10, 10, 21)):
        self.alphas = alphas
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)
        cv_scores = []
        for alpha in self.alphas:
            mse_scores = []
            for i in range(n_samples):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i, axis=0)
                X_val = X[i, :].reshape(1, -1)
                y_val = y[i]
                w = np.linalg.solve(X_train.T @ X_train + alpha * np.eye(n_features), X_train.T @ y_train)
                y_pred = self.sigmoid(X_val @ w + self.intercept_)
                mse_scores.append((y_val - y_pred) ** 2)
            cv_scores.append(np.mean(mse_scores))
        self.alpha_ = self.alphas[np.argmin(cv_scores)]
        self.coef_ = np.linalg.solve(X.T @ X + self.alpha_ * np.eye(n_features), X.T @ y)
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        z = X @ self.coef_ + self.intercept_
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)