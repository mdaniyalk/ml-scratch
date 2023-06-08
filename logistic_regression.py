import numpy as np 
from tqdm import tqdm

class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters:
    - learning_rate (float): The learning rate for gradient descent optimization.
    - num_iterations (int): The number of iterations for training.

    This class implements a Logistic Regression classifier. It performs binary classification by fitting a logistic function to the input data.

    The class has three main methods: `fit`, `predict`, and `sigmoid`. 
    The `fit` method trains the classifier using gradient descent optimization. 
    The `predict` method makes predictions on new data using the trained model. 
    The `sigmoid` method calculates the sigmoid function, which is used to map the linear model's output to a probability value between 0 and 1.

    Example usage:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the LogisticRegression object.

        Parameters:
        - learning_rate (float): The learning rate for gradient descent optimization.
        - num_iterations (int): The number of iterations for training.

        This method initializes the LogisticRegression object with the specified learning rate and number of iterations. The weights and bias are set to None initially.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Logistic Regression model to the training data.

        Parameters:
        - X (ndarray): The input features.
        - y (ndarray): The target labels.

        This method trains the Logistic Regression model by performing gradient descent optimization. It iteratively updates the weights and bias based on the calculated gradients.

        The method initializes the weights to zeros and the bias to zero. It then performs gradient descent for the specified number of iterations. For each iteration, it calculates the linear model's output, applies the sigmoid function to obtain predicted probabilities, computes the gradients, and updates the weights and bias using the gradients and learning rate.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in tqdm(range(self.num_iterations)):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the class labels for new data.

        Parameters:
        - X (ndarray): The input features of the new data.

        Returns:
        - y_pred (ndarray): The predicted class labels.

        This method predicts the class labels for new data using the trained model. It calculates the linear model's output, applies the sigmoid function, and then converts the probabilities to binary class labels based on a threshold of 0.5.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)

    def sigmoid(self, x):
        """
        Calculate the sigmoid function.

        Parameters:
        - x (ndarray): The input to the sigmoid function.

        Returns:
        - output (ndarray): The sigmoid function output.

        This method calculates the sigmoid function, which maps the linear model's output to a probability value between 0 and 1. It applies the formula: output = 1 / (1 + exp(-x)).
        """
        return 1 / (1 + np.exp(-x))
