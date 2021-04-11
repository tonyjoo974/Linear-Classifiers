"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, input_dim: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5
        self.w = np.random.rand(input_dim)  

    def sigmoid(self, z: float) -> float:
        """Sigmoid function.

        Parameters:
            x: the input

        Returns:
            the sigmoid of the input
        """
        # check if overflow from value will exceed floating point max at exp()
        # from https://stackoverflow.com/questions/48540589/sigmoid-runtimewarning-overflow-encountered-in-exp
        # user kazemakase
        if -z > np.log(np.finfo(type(z)).max):
          return 0.0
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        max_batch_size = 32
        num_entries = y_train.shape[0]
        indList = np.arange(num_entries)
        for epoch in range(self.epochs):
          curr_ind = 0
          decay_rate = 0.7
          decay_lr = self.lr / (1 + decay_rate * epoch)
          # shuffle the indexes
          np.random.shuffle(indList)
          while curr_ind < num_entries:
            batch_size = max_batch_size
            entries_left = num_entries - curr_ind
            grad = 0
            if (int(entries_left / max_batch_size) == 0):
              batch_size = entries_left 
            # train each batch
            for j in range(batch_size):
              i = indList[curr_ind]
              x_i = X_train[i]
              y_i = y_train[i]
              if y_i == 0:
                y_i = -1
              grad += -1 * self.sigmoid(-y_i * np.dot(self.w, x_i)) * y_i * x_i
              curr_ind += 1  
            # update the weights after batch update
            #grad /= batch_size
            self.w -= decay_lr * grad
            
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        Y_test = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
          output = self.sigmoid(np.dot(self.w, X_test[i]))
          if output >= self.threshold:
            Y_test[i] = 1
          else:
            Y_test[i] = 0
        return Y_test