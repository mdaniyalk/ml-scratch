import numpy as np 
from tqdm import tqdm
import pandas as pd 
from joblib import Parallel, delayed


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
    - right: Link to the right child node.
    - left: Link to the left child node.
    - column: Column index used for splitting criteria.
    - threshold: Threshold value used for splitting criteria.
    - probas: Probability for objects inside the node to belong to each of the given classes.
    - depth: Depth of the node in the tree.
    - is_terminal: Indicates whether the node is a terminal node or not.

    The `Node` class represents a node in a decision tree. Each node can have links to its right and left child nodes. It also stores information derived from the splitting criteria, such as the column index (`column`) and threshold value (`threshold`).

    The `probas` attribute represents the probability for objects inside the node to belong to each of the given classes. The `depth` attribute indicates the depth of the node in the tree. The `is_terminal` attribute is a boolean flag that indicates whether the node is a terminal node or not.

    Example usage:
    node = Node()
    node.column = 0
    node.threshold = 0.5
    node.is_terminal = True
    """
    
    def __init__(self):
        self.right = None
        self.left = None
        self.column = None
        self.threshold = None
        self.probas = None
        self.depth = None
        self.is_terminal = False



class DecisionTreeClassifier:
    """
    Decision tree classifier.

    Parameters:
    - max_depth (int): The maximum depth of the decision tree. Defaults to 3.
    - min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Defaults to 1.
    - min_samples_split (int): The minimum number of samples required to split an internal node. Defaults to 2.


    This class represents a decision tree classifier. It can be used for binary or multi-class classification problems. The decision tree is built using the provided training data and can be used to make predictions on new data.

    The decision tree is constructed based on the specified parameters, such as the maximum depth, minimum samples required at a leaf node, and minimum samples required to split an internal node.

    The class provides several methods for various functionalities of the decision tree classifier.

    Example usage:
    dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    """

    def __init__(self, max_depth=3, min_samples_leaf=1, min_samples_split=2):
        """
        Initializes a decision tree classifier.

        Parameters:
        - max_depth (int): The maximum depth of the decision tree. Defaults to 3.
        - min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Defaults to 1.
        - min_samples_split (int): The minimum number of samples required to split an internal node. Defaults to 2.

        This class represents a decision tree classifier. It uses the Gini impurity as the criterion for splitting nodes. The tree is built recursively, starting from the root node.

        The class provides methods for training the tree on input data and making predictions on new data.

        Example usage:
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        """

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

        self.classes = None
        self.verbosity = None
        self.Tree = None


    def nodeProbas(self, y: np.ndarray, sample_weight: np.ndarray = None) -> np.ndarray:
        """
        Calculates the probability of each class in a given node.

        Parameters:
        - y (ndarray): The target variable values for the node.
        - sample_weight (ndarray): The weights assigned to the samples. Defaults to None.

        Returns:
        - probas (ndarray): The array of class probabilities for the node.

        This method calculates the probability of each class in a given node. It accepts the target variable values `y` and an optional array of sample weights `sample_weight`. If `sample_weight` is not provided, equal weights are assigned to all samples.

        The method iterates over each class in the `classes` attribute and calculates the probability by summing the sample weights for that class and dividing by the total sum of sample weights. The probabilities for all classes are returned as an array.

        Example usage:
        probas = dt_classifier.nodeProbas(y)
        """

        probas = []
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        
        assert y.shape == sample_weight.shape, f"y {y.shape} and sample_weight {sample_weight.shape} must have the same shape"

        for one_class in self.classes:
            mask = y == one_class
            proba = np.sum(sample_weight[mask]) / np.sum(sample_weight)
            probas.append(proba)
        return np.asarray(probas)


    def gini(self, probas: np.ndarray) -> float:
        """
        Calculates the Gini impurity criterion.

        Parameters:
        - probas (ndarray): The array of class probabilities.

        Returns:
        - impurity (float): The Gini impurity.

        This method calculates the Gini impurity criterion based on the given class probabilities. It accepts an array of class probabilities and returns the calculated Gini impurity.

        The Gini impurity is calculated as 1 minus the sum of squared probabilities for all classes. The lower the impurity, the better the split.

        Example usage:
        impurity = dt_classifier.gini(probas)
        """

        return 1 - np.sum(probas**2)


    def calcImpurity(self, y, sample_weight=None):
        """
        Calculates the impurity of a node.

        Parameters:
        - y (ndarray): The target variable values for the node.
        - sample_weight (ndarray): The weights assigned to the samples. Defaults to None.

        Returns:
        - impurity (float): The impurity of the node.

        This method calculates the impurity of a node based on the target variable values `y`. It accepts an optional array of sample weights `sample_weight`, which defaults to equal weights if not provided.

        The method first calculates the class probabilities using the `nodeProbas()` method. It then passes these probabilities to the `gini()` method to calculate the impurity.

        Example usage:
        impurity = dt_classifier.calcImpurity(y)
        """

        if sample_weight is None:
            sample_weight = np.ones_like(y)
        return self.gini(self.nodeProbas(y, sample_weight=sample_weight))
    

    def calcInfoGain(self, impurityLeft: float, y_left: np.ndarray, impurityRight: float, y_right: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the information gain of a split.

        Parameters:
        - impurityLeft (float): The impurity of the left child node.
        - y_left (ndarray): The target variable values for the left child node.
        - impurityRight (float): The impurity of the right child node.
        - y_right (ndarray): The target variable values for the right child node.
        - y (ndarray): The target variable values for the current node.

        Returns:
        - infogain (float): The information gain of the split.

        This method calculates the information gain of a split based on the impurities and target variable values of the left and right child nodes, as well as the current node.

        The information gain is calculated as the weighted sum of impurities, where the weights are the proportions of samples in each child node relative to the current node.

        Example usage:
        infogain = dt_classifier.calcInfoGain(impurityLeft, y_left, impurityRight, y_right, y)
        """
        
        infogain = (impurityLeft * y_left.shape[0] / y.shape[0]) + (impurityRight * y_right.shape[0] / y.shape[0])
        return infogain

    
    def calcBestSplit(self, X, y, sample_weight=None):
        """
        Calculates the best possible split for the current node of the decision tree.

        Parameters:
        - X (ndarray): The feature matrix for the current node.
        - y (ndarray): The target variable values for the current node.
        - sample_weight (ndarray): The weights assigned to the samples. Defaults to None.

        Returns:
        - bestSplitCol (int): The index of the column with the best split.
        - bestThresh (float): The threshold value for the best split.
        - x_left (ndarray): The feature matrix for the left child node.
        - y_left (ndarray): The target variable values for the left child node.
        - x_right (ndarray): The feature matrix for the right child node.
        - y_right (ndarray): The target variable values for the right child node.
        - sample_weight_right (ndarray): The weights assigned to the samples in the right child node.
        - sample_weight_left (ndarray): The weights assigned to the samples in the left child node.

        This method calculates the best possible split for the current node of the decision tree based on the feature matrix `X` and target variable values `y`. It accepts an optional array of sample weights `sample_weight`, which defaults to equal weights if not provided.

        The method iterates over each column in `X` and each unique value in that column. For each combination, it calculates the information gain using the `calcInfoGain()` method and compares it to the best information gain found so far. The best split column, threshold value, and child node data are returned.

        Example usage:
        bestSplitCol, bestThresh, x_left, y_left, x_right, y_right, sample_weight_right, sample_weight_left = dt_classifier.calcBestSplit(X, y)
        """
        
        bestSplitCol = None
        bestThresh = None
        bestInfoGain = -999
        
        impurityBefore = self.calcImpurity(y, sample_weight)

        if sample_weight is None:
            sample_weight = np.ones_like(y)

        if sample_weight is not None:
            total_weight = np.sum(sample_weight)
        else:
            total_weight = None

        # for each column in X
        if self.verbosity > 0:
            col_range = tqdm(range(X.shape[1]))
        else :
            col_range = range(X.shape[1])
        for col in col_range:
            x_col = X[:, col]
            
            # for each value in the column
            for x_i in np.unique(np.round(np.unique(x_col), 2)):
                threshold = x_i
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]
                sample_weight_right = sample_weight[x_col > threshold]
                sample_weight_left = sample_weight[x_col <= threshold]
                
                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue

                # Parallelize
                def calcImpurity_tmp(y, sample_weight):
                    return self.calcImpurity(y, sample_weight)
                tmp = [[y_right, sample_weight_right], [y_left, sample_weight_left]]
                impurity_tmp = Parallel(n_jobs=-1)(delayed(calcImpurity_tmp)(y, sample_weight) for y, sample_weight in tmp)
                impurityRight = impurity_tmp[0]
                impurityLeft = impurity_tmp[1]
                # calculate information gain
                infoGain = impurityBefore
                infoGain -= self.calcInfoGain(impurityLeft, y_left, impurityRight, y_right, y)
                
                # is this infoGain better then all other?
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain
                    
        
        # if we still didn't find the split
        if bestInfoGain == -999:
            return None, None, None, None, None, None, None, None
        
        # making the best split
        
        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]
        sample_weight_left, sample_weight_right = sample_weight[x_col <= bestThresh], sample_weight[x_col > bestThresh]
        
        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right, sample_weight_right, sample_weight_left
                
          
    def buildDT(self, X, y, node, sample_weight):
        """
        Recursively builds the decision tree from top to bottom.

        Parameters:
        - X (ndarray): The feature matrix for the current node.
        - y (ndarray): The target variable values for the current node.
        - node (Node): The current node of the decision tree.
        - sample_weight (ndarray): The weights assigned to the samples.

        This method recursively builds the decision tree from the top to the bottom. It accepts the feature matrix `X`, target variable values `y`, current node `node`, and an array of sample weights `sample_weight`.

        The method first checks for the terminal conditions, such as reaching the maximum depth, having too few samples to split, or having only one unique class in the target variable values. If any of these conditions are met, the current node is marked as a terminal node.

        If the terminal conditions are not met, the method calls the `calcBestSplit()` method to determine the best possible split for the current node. It then assigns the split column, threshold value, and creates the left and right child nodes.

        Finally, the method recursively calls itself on the left and right child nodes to continue building the decision tree.

        Example usage:
        dt_classifier.buildDT(X, y, node, sample_weight)
        """
        
        # checking for the terminal conditions
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return
            
        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return
            
        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return
        
        # calculating current split
        splitCol, thresh, x_left, y_left, x_right, y_right, sample_weight_right, sample_weight_left = self.calcBestSplit(X, y, sample_weight)
        assert y_right.shape == sample_weight_right.shape and y_left.shape == sample_weight_left.shape, f"y_right {y_right.shape} and sample_weight_right {sample_weight_right.shape} must have the same shape\ny_left {y_left.shape} and sample_weight_left {sample_weight_left.shape} must have the same shape"
        if splitCol is None:
            node.is_terminal = True
            
        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return
        
        node.column = splitCol
        node.threshold = thresh
        
        # creating left and right child nodes
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probas = self.nodeProbas(y_left, sample_weight_left)
        
        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probas = self.nodeProbas(y_right, sample_weight_right)
        
        # splitting recursively
        # self.buildDT(x_right, y_right, node.right, sample_weight_right)
        # self.buildDT(x_left, y_left, node.left, sample_weight_left)
        dt_tmp = [[x_right, y_right, node.right, sample_weight_right], [x_left, y_left, node.left, sample_weight_left]]
        Parallel(n_jobs=-1)(delayed(self.buildDT)(_x, _y, _node, _sample_weight) for _x, _y, _node, _sample_weight in dt_tmp)
        
     
    def fit(self, X, y, sample_weight=None, verbose=1):
        """
        Builds the decision tree based on the input data and labels.

        Parameters:
        - X (ndarray): The input data.
        - y (ndarray): The target variable values.
        - sample_weight (ndarray): The weights assigned to the samples. Defaults to None.
        - verbose (int): The level of verbosity. Defaults to 1.

        This method builds the decision tree based on the provided input data `X` and labels `y`. It accepts an optional array of sample weights `sample_weight` and a verbosity level `verbose`.

        The method first converts the input data and labels into NumPy arrays. It then initializes the `self.classes` attribute with the unique classes in the labels. If `sample_weight` is not provided, equal weights are assigned to all samples.

        The root node of the decision tree is created, and its depth and class probabilities are set. The `buildDT()` method is called recursively to build the tree.

        Example usage:
        dt_classifier.fit(X_train, y_train, sample_weight=weights, verbose=2)
        """

        X = np.asarray(X)
        y = np.asarray(y)
        self.verbosity = verbose

        # save the unique classes in the labels
        self.classes = np.unique(y)
        
        # if sample weights are not provided, use an array of ones
        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])

        assert y.shape == sample_weight.shape, "y and sample_weight must have the same shape"

        # create the root node
        self.Tree = Node()
        self.Tree.depth = 0
        self.Tree.probas = self.nodeProbas(y, sample_weight)
        
        # build the tree recursively
        self.buildDT(X, y, self.Tree, sample_weight)
    
    
    def predictSample(self, x, node):
        """
        Passes one object through the decision tree and returns the probability of it belonging to each class.

        Parameters:
        - x (ndarray): The input data point.
        - node (Node): The current node in the decision tree.

        Returns:
        - probas (ndarray): The array of class probabilities for the input data point.

        This method passes a single input data point `x` through the decision tree and returns the probability of it belonging to each class. It accepts the current node `node` as a parameter.

        If the terminal node of the tree or a node without a threshold is reached, the method returns the class probabilities stored in the node. Otherwise, it recursively calls itself on the left or right child node based on the threshold comparison.

        Example usage:
        probas = dt_classifier.predictSample(x, node)
        """
    
        # if we have reached the terminal node of the tree
        if node.is_terminal  or node.threshold is None:
            return node.probas
        
        if x[node.column] > node.threshold:
            probas = self.predictSample(x, node.right)
        else:
            probas = self.predictSample(x, node.left)
            
        return probas
        
        
    def predict(self, X):
        """
        Returns the predicted labels for the input data.

        Parameters:
        - X (ndarray or DataFrame): The input data.

        Returns:
        - predictions (ndarray): The predicted labels.

        This method returns the predicted labels for the input data `X`. It accepts either a NumPy array or a pandas DataFrame as the input.

        The method iterates over each data point in `X` and calls the `predictSample()` method to obtain the class probabilities. The predicted label is then determined by selecting the class with the highest probability.

        Example usage:
        predictions = dt_classifier.predict(X_test)
        """
        
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
            
        predictions = []
        for x in X:
            pred = np.argmax(self.predictSample(x, self.Tree))
            predictions.append(pred)
        
        return np.asarray(predictions)

    def predict_proba(self, X):
        """
        Returns the predicted class probabilities for the input data.

        Parameters:
        - X (ndarray or DataFrame): The input data.

        Returns:
        - predictions (ndarray): The predicted class probabilities.

        This method returns the predicted class probabilities for the input data `X`. It accepts either a NumPy array or a pandas DataFrame as the input.

        The method iterates over each data point in `X` and calls the `predictSample()` method to obtain the class probabilities.

        Example usage:
        proba_predictions = dt_classifier.predict_proba(X_test)
        """
        
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
            
        predictions = []
        for x in X:
            pred = self.predictSample(x, self.Tree)
            predictions.append(pred)
        
        return np.asarray(predictions)