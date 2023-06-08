import numpy as np

import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Random Forest Classifier.

    This class implements a random forest classifier, which is an ensemble learning method that combines multiple decision trees to make predictions.

    Parameters:
    - n_estimators (int): The number of decision trees in the random forest.
    - max_depth (int): The maximum depth of each decision tree. If None, the decision trees will be grown until all leaves are pure or until all leaves contain min_samples_split samples.
    - min_samples_split (int): The minimum number of samples required to split an internal node in each decision tree.

    Attributes:
    - n_estimators (int): The number of decision trees in the random forest.
    - max_depth (int): The maximum depth of each decision tree.
    - min_samples_split (int): The minimum number of samples required to split an internal node in each decision tree.
    - estimators (list): A list of decision trees in the random forest.

    Methods:
    - fit(X, y): Fit the random forest to the training data.
    - predict(X): Make predictions for new data.

    Example usage:
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Initialize the Random Forest Classifier.

        Parameters:
        - n_estimators (int): The number of decision trees in the random forest.
        - max_depth (int): The maximum depth of each decision tree. If None, the decision trees will be grown until all leaves are pure or until all leaves contain min_samples_split samples.
        - min_samples_split (int): The minimum number of samples required to split an internal node in each decision tree.

        This method initializes the random forest classifier with the specified parameters. It sets the number of decision trees, maximum depth, minimum samples to split, and creates an empty list to store the decision tree estimators.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators = []

    def fit(self, X, y):
        """
        Fit the random forest to the training data.

        Parameters:
        - X (ndarray): The input features of the training data.
        - y (ndarray): The target values of the training data.

        This method fits the random forest to the provided training data. It creates `n_estimators` decision trees, each trained on a bootstrap sample with replacement. Each decision tree is created using the `DecisionTreeClassifier` class with the specified `max_depth` and `min_samples_split` parameters. The trained decision trees are added to the `estimators` list.
        """
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
        """
        Make predictions for new data.

        Parameters:
        - X (ndarray): The input features of the new data.

        Returns:
        - y_pred (ndarray): The predicted target values for the new data.

        This method makes predictions for the provided new data by aggregating the predictions of each decision tree in the random forest. The final prediction is determined by taking the majority vote of the predictions from all decision trees. If the number of positive predictions is equal to or greater than half the number of estimators, the target value is assigned as 1; otherwise, it is assigned as 0.
        """
        y_pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            y_pred += estimator.predict(X)
        return (y_pred >= (len(self.estimators) / 2)).astype(int)
