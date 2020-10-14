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
# import random
from scipy.spatial import distance


class testKNN():

    def euclidean_distance(self, a, b):
        """
            Description:
                Computes the Euclidean distance between two 1-D arrays.
            Parameters: 
                Array like, input array
                point a, point b
            Returns: 
                The euclidean distance between points a and b: double
        """
        return distance.euclidean(a,b)


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
        # Initializing the training data and labels
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
            # Predict the closest point label for the testing point
            label = self.closest_point(row) 
            # Append the label into the predictions list
            predictions.append(label)

        return predictions

    def closest_point(self, row):
        """
            Description:
                Iterates through the training points X_train[i] and
                retrieves the closest point label to the testing point 
                X_test[i](row[i])
            Parameters:
                X_test[i](row[i]): The testing point

            Returns: 
                The clossest point training label y_train[i] 
        """
        # Defining the shortest distance to be the first row in training data
        shortest_distance = self.euclidean_distance(row, self.X_train[0])
        # The index representing the label.
        shorted_index = 0
        # Iterating through the training data to find the 
        # shortest_distance/point to the testing point and update the
        # distance and index to the testing point.
        for i in range(1, len(self.X_train)):
            distance = self.euclidean_distance(row, self.X_train[i])
            if distance < shortest_distance:
                shortest_distance = distance
                shorted_index = i

        return self.y_train[shorted_index]



