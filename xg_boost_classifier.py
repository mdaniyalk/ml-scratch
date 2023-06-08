import numpy as np 
import DecisionTreeClassifier

class XGBoostClassifier:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0, colsample=1.0, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample = colsample
        self.reg_lambda = reg_lambda
        self.trees = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._f = np.zeros(n_samples)
        
        for i in range(n_classes):
            y_binary = np.where(y == self._classes[i], 1, -1)
            f = np.zeros(n_samples)
            for j in range(self.n_estimators):
                tree = self._fit_tree(X, y_binary, f)
                f += self.learning_rate * tree.predict(X)
                self.trees.append(tree)
            self._f += f
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self._classes[np.argmax(probs, axis=1)]
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self._classes)
        probs = np.zeros((n_samples, n_classes))
        for tree in self.trees:
            probs += tree.predict_proba(X)
        probs /= self.n_estimators
        return probs
    
    def _fit_tree(self, X, y, f):
        n_samples, n_features = X.shape
        idxs = np.random.choice(n_samples, int(self.subsample * n_samples), replace=False)
        cols = np.random.choice(n_features, int(self.colsample * n_features), replace=False)
        X_subset = X[idxs][:, cols]
        y_subset = y[idxs]
        f_subset = f[idxs]
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(X_subset, y_subset, sample_weight=self._get_sample_weights(y_subset, f_subset))
        return tree
    
    def _get_sample_weights(self, y, f):
        # Gradient Boosting sample weights calculation
        p = self._sigmoid(f)
        sample_weights = p * (1 - p)
        sample_weights = np.clip(sample_weights, 1e-6, 1 - 1e-6)
        sample_weights = np.where(y == 1, 1 / sample_weights, 1 / (1 - sample_weights))
        sample_weights /= np.mean(sample_weights)
        return sample_weights
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))