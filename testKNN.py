"""
    Class testKNN:
    A scrappy version of KNN classifier

    >>> Methods:
            fit(self, X_train, y_train)
            >>> Parameters:
                >>> X_train - Training features: <ND list>
                >>> y_train - Training labels: <ND list> 

            predict(self, X_test)
            >>> Parameters:
                >>> X_test - Test features to predict the labels: <ND list>
"""
import random

class testKNN():
    def fit(self, X_train, y_train):
        """
            Description: 
                Fit the model using X as training data and y as target values. 
            
            Parameters:    
                X_train: 
                    Training features/data {array-like, 
                    sparse matrix, BallTree, KDTree} Training data. 
                    If array or matrix, shape [n_samples, n_features], 
                    or [n_samples, n_samples] if metric=’precomputed’.

                y_train: 
                    Training labels/targets {array-like, sparse matrix}
                    Target values of shape = [n_samples] or 
                    [n_samples, n_outputs]

            Return: None
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
            Description: Predict the class labels for the provided data. 

            Parameters:    
                X_test: Training features/data 
                array-like of shape (n_queries, n_features), 
                or (n_queries, n_indexed) if metric == ‘precomputed’

            Return: 
                Predictions ndarray of shape (n_queries,) or 
                (n_queries, n_outputs) Class labels for each data sample.
        """
        predictions = list()

        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)

        return predictions



