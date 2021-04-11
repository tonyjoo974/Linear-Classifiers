"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, dim: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        # set weights to either multi class or binary
        if self.n_class > 2:
          self.w = np.random.rand(n_class, dim)
        else:
          self.w = np.random.rand(dim)  

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray, n: int) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C
            n: total number of training data

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        w_grad = np.zeros(self.w.shape)
        batch_size = y_train.shape[0]
        if self.n_class == 2:
          for i in range(batch_size):
            x_i = X_train[i]
            # change y label 1/0 to 1/-1
            y_i = 1
            if y_train[i] == 0:
              y_i = -1
            # sum up the gradients
            w_grad += (self.reg_const / n) * self.w
            if y_i * np.dot(self.w, x_i) < 1:
              w_grad -= y_i * x_i
          return w_grad
        else:
          for i in range(batch_size):
            x_i = X_train[i]
            y_i = y_train[i]
            # update gradient
            for c in range(self.n_class):
              w_grad[c] += self.reg_const * self.w[c] / n
              if c != y_i and (np.dot(self.w[y_i], x_i) - np.dot(self.w[c], x_i) < 1):
                w_grad[y_i] -= x_i
                w_grad[c] += x_i
          return w_grad


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        max_batch_size = 32
        num_entries = y_train.shape[0]
        # randomize the inputs for each epoch 
        indList = np.arange(num_entries)
        for epoch in range(self.epochs):
          ind = 0
          decay_rate = 0.5
          decay_lr = self.alpha / (1 + decay_rate * epoch)
          if epoch % 10 == 0:
            print("epoch:", epoch)
          np.random.shuffle(indList)
          while ind < num_entries:
            batch_size = max_batch_size
            entries_left = num_entries - ind
            if (int(entries_left / max_batch_size) == 0):
              batch_size = entries_left 
            # get gradient of batch
            X_train_batch = X_train[indList[ind:ind+batch_size]]
            y_train_batch = y_train[indList[ind:ind+batch_size]]
            # calculate gradient
            grad = self.calc_gradient(X_train_batch, y_train_batch, num_entries)
            self.w -= (decay_lr * grad)
            ind += batch_size
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
        predicts = []
        for i in range(X_test.shape[0]):
          if self.n_class > 2:  # multiclass classification
            outputs = np.zeros(self.n_class)
            for c in range(self.n_class):
              outputs[c] = np.dot(self.w[c], X_test[i])
            predicts.append(int(np.argmax(outputs)))
          else:  # binary classification
            output = np.dot(self.w, X_test[i])
            if output >= 0:
              predicts = 1
            else:
              predicts = 0
        return predicts