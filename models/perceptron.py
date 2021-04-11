"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N = X_train.shape[0]
        dim = X_train.shape[1]
        self.w = np.random.rand(self.n_class, dim)

        for each_epoch in range(self.epochs):
            if each_epoch % 10 == 0:
              print("epoch: ", each_epoch)
            decay_rate = 0.5
            decay_lr = self.lr / (1 + decay_rate * each_epoch)
            for i, x in enumerate(X_train):
                y_predict = np.argmax(np.dot(self.w, x))
                # print(y_predict)
                if y_train[i] != y_predict:
                    # print("y_train: ", y_train[i])
                    # print("y_predict: ", y_predict)
                    for c in range(self.n_class):
                        if np.dot(self.w[c], x) > np.dot(self.w[y_train[i]], x):
                            # print("w[y_train[i]]", self.w[y_train[i]])
                            # print("w[c]", self.w[c])
                            self.w[y_train[i]] += decay_lr * x
                            self.w[c] -= decay_lr * x
            decay_lr -= each_epoch*0.01
          
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
        # TODO: implement me
        test_labels = []
        for x in X_test:
            y_predict = np.argmax(np.dot(self.w, x))
            test_labels.append(int(y_predict))
          
        return test_labels