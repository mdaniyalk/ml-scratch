import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

class StackingClassifier:
    """
    Stacking Classifier implementation.

    This class implements a stacking classifier, which combines multiple base classifiers and a meta-classifier to make predictions.

    Parameters:
    - base_classifiers (list): List of base classifiers.
    - meta_classifier (object): Meta-classifier for combining the predictions of base classifiers.

    Example usage:
    base_clf1 = RandomForestClassifier()
    base_clf2 = GradientBoostingClassifier()
    meta_clf = LogisticRegression()
    stacking_clf = StackingClassifier([base_clf1, base_clf2], meta_clf)
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    """

    def __init__(self, base_classifiers, meta_classifier):
        """
        Initializes a StackingClassifier instance.

        Args:
        - base_classifiers (list): List of base classifiers.
        - meta_classifier (object): Meta-classifier for combining the predictions of base classifiers.
        """
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
    
    def fit(self, X, y):
        """
        Fits the stacking classifier to the training data.

        Args:
        - X (ndarray): The training input samples.
        - y (ndarray): The target values.

        This method fits the base classifiers to the training data and generates meta-features by making predictions on the training data using each base classifier. The meta-features are then used to fit the meta-classifier.

        Note: This method uses parallel processing with joblib.Parallel and joblib.delayed for efficient computation.

        Example usage:
        stacking_clf.fit(X_train, y_train)
        """
        meta_features = Parallel(n_jobs=-1)(delayed(self._fit_base_classifier)(clf, X, y) for clf in tqdm(self.base_classifiers, desc='Base Classifiers'))
        meta_features = np.hstack(meta_features)
        self.meta_classifier.fit(meta_features, y)
    
    def predict(self, X):
        """
        Predicts the target values for the given input samples.

        Args:
        - X (ndarray): The input samples for prediction.

        Returns:
        - y_pred (ndarray): The predicted target values.

        This method generates meta-features by making predictions on the input samples using each base classifier. The meta-features are then used to predict the target values using the meta-classifier.

        Note: This method uses parallel processing with joblib.Parallel and joblib.delayed for efficient computation.

        Example usage:
        y_pred = stacking_clf.predict(X_test)
        """
        meta_features = Parallel(n_jobs=-1)(delayed(clf.predict_proba)(X) for clf in tqdm(self.base_classifiers, desc='Base Classifiers'))
        meta_features = np.hstack(meta_features)
        return self.meta_classifier.predict(meta_features)
    
    def _fit_base_classifier(self, clf, X, y):
        """
        Fits a base classifier to the training data and returns its predicted probabilities.

        Args:
        - clf (object): The base classifier.
        - X (ndarray): The training input samples.
        - y (ndarray): The target values.

        Returns:
        - y_pred_proba (ndarray): The predicted probabilities of the base classifier.

        This method fits the base classifier to the training data and returns the predicted probabilities for the training samples.

        Example usage:
        y_pred_proba = self._fit_base_classifier(clf, X_train, y_train)
        """
        clf.fit(X, y)
        return clf.predict_proba(X)
