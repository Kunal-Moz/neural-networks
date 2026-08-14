#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


class CrossEntropyCost:
    """Categorical cross-entropy, paired with softmax output."""
    @staticmethod
    def loss(a, y):
        return -np.sum(y * np.log(a + 1e-9*np.ones(np.shape(a)))) / np.shape(y)[0]
    # np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y):
        return (a - y) / y.shape[0]      # softmax + CE shortcut


class MSECost:
    """Mean squared error, paired with softmax output here."""
    @staticmethod
    def loss(a, y):
        return np.sum((a - y)**2) / (2 * y.shape[0])

    @staticmethod
    def delta(a, y):
        return (a - y) / y.shape[0]      # assumes linear/identity output



class NeuralNet:

    def __init__(self, input_size = 784, ouptut_size = 10, layers=[64], lr=0.1, activation='relu', cost=CrossEntropyCost):
        """ Initialize the class:
            -- layers   : a list that contains the number of neurons in each hidden layer.
            -- lr       : Learning rate 
            -- cost     : Cost function used. 
        """
        self.layers = [input_size] + layers + [ouptut_size]
        # np.random.seed(42)
        self.num_layers = len(self.layers)
        self.activation_func = activation
        self.lr = lr
        self.cost = cost
        self.b = [np.zeros(y) for y in self.layers[1:] ]
        if activation == 'relu':
            self.W = [np.random.randn(x, y)*np.sqrt(2/x) for x, y in zip(self.layers[:-1], self.layers[1:]) ]
        else:
            self.W = [np.random.randn(x, y)*np.sqrt(2/x) for x, y in zip(self.layers[:-1], self.layers[1:]) ]    

 
    @staticmethod
    def _relu(z):
        """ The ReLU activation function"""
        return np.maximum(0, z)

    @staticmethod
    def _sigmoid(z):
        """The Sigmoid activation function."""
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def out_val(y, classes = 10):
        out = np.zeros((np.size(y), classes))
        out[np.arange(y.size), y] = 1
        return out


    def _act_deriv(self,z):
        if self.activation_func == 'relu':
            return (z > 0).astype(z.dtype)
        elif self.activation_func == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        else:
            return None
        

    def forward(self, x):
        self.Zs, self.activations = [], [x]
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[i], self.W[i]) + self.b[i]     #### z = x.W + b 
            self.Zs.append(z)
            if self.activation_func == 'relu':
                self.activations.append(self._relu(z) if i < len(self.W) - 1 else self.softmax(z))    #### a_i = ReLu(z)
            elif self.activation_func == 'sigmoid':    
                self.activations.append(self._sigmoid(z) if i < len(self.W) - 1 else self.softmax(z))   #### a_i = sigmoid(z)
            else:
                print("Activation function unknown")
                None
        return self.activations

    def backpropagation(self, yb):
        dz = (self.cost).delta(self.activations[-1], yb)
        for l in range(self.num_layers - 2, -1, -1):
            dW = np.dot(np.transpose(self.activations[l]),dz)
            db = dz.sum(axis = 0)
            if l > 0:
                zs = self.Zs[l-1]
                dz = np.dot(dz, np.transpose(self.W[l]))*self._act_deriv(zs) 
            self.W[l] -= self.lr*dW
            self.b[l] -= self.lr*db

    def SGD(self, X, y, X_val=None, y_val=None, epochs=20, batch_size=64):
        Y = self.out_val(y)
        loss, accuracy = [] , []
        for epoch in range(epochs):
            idx = np.random.permutation(X.shape[0])
            Xs, Ys = X[idx], Y[idx]
            epoch_loss = 0.0
            for i in range(0, Xs.shape[0], batch_size):
                yb = Ys[i:i+batch_size]
                out = self.forward(Xs[i:i+batch_size])
                epoch_loss += self.cost.loss(out[-1], yb) * yb.shape[0]
                self.backpropagation(yb)
            epoch_loss /= X.shape[0]
            # msg = f"epoch {epoch+1}: loss {epoch_loss:.4f}"
            if X_val is not None:
                acc = (self.predict(X_val) == y_val).mean()
                # msg += f" | val acc {acc:.4f}"
                accuracy.append(acc)
            loss.append(epoch_loss)
            # print(msg)
            print(f"epoch {epoch+1}: loss {loss[-1]:.4f}", 
              f"| val acc {accuracy[-1]:.4f}" if X_val is not None else "")
        return loss, accuracy

    def predict(self, X):
        return self.forward(X)[-1].argmax(axis=1)

    
    