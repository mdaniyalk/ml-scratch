import numpy as np

import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        for i in range(self.n_estimators):
            # Bootstrap sample with replacement
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            # Create decision tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot)
            # Add tree to list of estimators
            self.estimators.append(tree)
        
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            y_pred += estimator.predict(X)
        return (y_pred >= (len(self.estimators) / 2)).astype(int)