import numpy as np 
from tqdm import tqdm

import DecisionTreeClassifier


class XGBoostClassifier:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.trees = []
   
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        class_counts = np.bincount(y)
        class_weights = n_samples / (len(class_counts) * class_counts)
        self._f = 1e4*class_weights[y]
        for i in range(n_classes):
            # y_binary = np.where(y == self._classes[i], 1, -1)
            y_binary = y
            f = self.learning_rate * self._f
            f = np.zeros(n_samples)
            for j in tqdm(range(self.n_estimators), desc=f"XGBoost Class {i+1}/{n_classes}"):
                tree = self._fit_tree(X, y_binary, f, j)
                out = tree.predict(X)
                f += (self.learning_rate * out)
                self.trees.append(tree)
            # self._f += f
            # self._f = (self.learning_rate * self._sigmoid(self._f))/((self.n_estimators))


    def predict(self, X):
        probs = self.predict_proba(X)
        # return self._classes[np.argmax(probs, axis=1)]
        # return probs
        return (probs >= ((self.n_estimators) / 2)).astype(int)
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self._classes)
        # probs = np.zeros((n_samples, n_classes))
        probs = np.zeros(n_samples)
        for tree in self.trees:
            probs += tree.predict(X)
        # probs /= self.n_estimators

        return probs

    def _fit_tree(self, X, y, f, j):
        n_samples, n_features = X.shape
        # n_subsample = int(self.subsample * n_samples * (j+1) / self.n_estimators)
        n_subsample = int(self.subsample * n_samples )
        idxs = np.random.choice(n_samples, n_subsample, replace=False)
        X_subset = X[idxs]
        y_subset = y[idxs]
        f_subset = f[idxs]
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(X_subset, y_subset, sample_weight=self._get_sample_weights(y_subset, f_subset), verbose=0)
        return tree
    
    # @jit(nopython=True)
    def _get_sample_weights(self,y: np.ndarray, f: np.ndarray) -> np.ndarray:
        # Gradient Boosting sample weights calculation
        p = self._sigmoid(f)
        sample_weights = p * (1 - p)
        sample_weights = np.clip(sample_weights, 1e-6, 1 - 1e-6)
        sample_weights = np.where(y == 1, 1 / sample_weights, 1 / (1 - sample_weights))
        sample_weights /= np.mean(sample_weights)
        return sample_weights
    
    # @jit(nopython=True)
    def _sigmoid(self,x: np.ndarray) -> np.ndarray:
        return np.exp(-np.logaddexp(0, -x))