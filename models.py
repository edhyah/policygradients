#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural network implementation.
Not intended for production use.
"""

import numpy as np

class LogisticPolicy():
    """Logistic policy class.

    Also known as sigmoid. Takes the form

        logistic(x) =  ________1________
                       1 + exp(-theta*x)

    where theta is a vector of weights.

    Let pi_theta be our logistic policy parameterized by theta. For ease of
    notation, pi will be used to denote pi_theta. Now, let pi(0|x) = logistic(x)
    be the probability of action 0. Then, pi(1|x) = 1 - pi(0|x). Mathematically:

        pi(0|x) = ________1________ = __exp(theta*x)__
                  1 + exp(-theta*x)   1 + exp(theta*x)

        pi(1|x) = 1 - pi(0|x) = _______1________
                                1 + exp(theta*x)

    Then, the policy gradients with respect to the policy parameter theta can be
    computed analytically, yielding the following:

        ∇ log pi(0|x) = ∇ log __exp(theta*x)__
                              1 + exp(theta*x)
                      = ∇ ( theta*x - log(1 + exp(theta*x)) )
                      = x - __x*exp(theta*x)__
                             1 + exp(theta*x)
                      = x - x * pi(0|x)

        ∇ log pi(1|x) = ∇ ( - log(1 + exp(theta*x)) )
                      = _ __x*exp(theta*x)__
                           1 + exp(theta*x)
                      = - x * pi(0|x)

    Attributes:
        theta: Policy parameter.
    """

    def __init__(self, theta):
        self.theta = theta

    def logistic(self, y):
        """Call the logistic function."""
        return 1 / (1 + np.exp(-y))

    def probs(self, x):
        """Returns binary probabilities given an input x."""
        y = x @ self.theta
        prob = self.logistic(y)
        return np.array([prob, 1-prob])

    def act(self, x):
        """Sample an action based on outputted probability distributions."""
        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)
        return action, probs[action]

    def grad_log_p(self, x):
        """Calculate gradients of policy given state x."""
        y = x @ self.theta
        grad_log_p0 = x - x * self.logistic(y)
        grad_log_p1 = - x * self.logistic(y)
        return grad_log_p0, grad_log_p1


class MultilayerPerceptron():
    """Multilayer perceptron class.

    Also known as MLP. An MLP with one hidden layer takes the form

        y' = dense(relu(dense(x)))

    where y' is the prediction and x is the input vector. An MLP with two hidden
    layers takes the form

        y' = dense(relu(dense(relu(dense(x)))))

    and so on, for MLPs with more hidden layers.

    We optimize the network's weights with respect to the mean squared error
    (L2 loss):

        loss = 1/N * Σ (y_true - y_pred)

    where y_true is a vector of ground truths and y_pred is a vector of
    predictions made by the network. N is the batch size, or number of examples
    we're optimizing over.

    Attributes:
        input_units: Number of input units.
        output_units: Number of output units.
        hidden_units: Number of hidden units.
        num_hidden: Number of hidden layers.
        lr: Learning rate.
    """

    def __init__(
            self,
            input_units,
            output_units,
            hidden_units=4,
            num_hidden=1,
            lr=0.1):
        if num_hidden < 1:
            raise ValueError("Must have a non-zero number of hidden layers.")
        self.num_hidden = num_hidden
        self.lr = lr
        self.input_layer = Dense(input_units, hidden_units)
        self.hidden_layers = \
                [Dense(hidden_units, hidden_units) for _ in range(num_hidden-1)]
        self.activations = [ReLU() for _ in range(num_hidden + 1)]
        self.output_layer = Dense(hidden_units, output_units)
        self.activation = ReLU()

    def predict(self, x):
        """
        Args:
            x: Input vector with dimensions (batch size) x (# of input units)
        Returns:
            Prediction of network (ie. forward pass).
        """
        y = self.input_layer.forward(x)
        y = self.activations[0].forward(y)
        for (i, hidden_layer) in enumerate(self.hidden_layers):
            y = hidden_layer.forward(y)
            y = self.activations[i+1].forward(y)
        y = self.output_layer.forward(y)
        return y

    def loss(self, y_pred, y_true):
        """
        Args:
            y_pred: Vector of predictions by network with dimensions
                    (batch size) x 1
            y_true: Vector of ground truths with dimensions (batch size) x 1
        Returns:
            MSE loss.
        """
        assert(y_pred.shape == y_true.shape)
        return np.mean((y_true - y_pred)**2)

    def update(self, y_pred, y_true):
        """Gradient update function.
        Args:
            y_pred: Vector of predictions by network with dimensions
                    (batch size) x 1
            y_true: Vector of ground truths with dimensions (batch size) x 1
        """
        assert(y_pred.shape == y_true.shape)

        gradient = -0.5*(y_true - y_pred)/y_pred.shape[0]
        gradient, gradient_W, gradient_b = self.output_layer.gradients(gradient)
        self.output_layer.W = self.output_layer.W - gradient_W * self.lr
        self.output_layer.b = self.output_layer.b - gradient_b * self.lr

        for (i, hidden_layer) in enumerate(self.hidden_layers):
            gradient = self.activations[num_hidden-i].gradients(gradient)
            gradient, gradient_W, gradient_b = hidden_layer.gradients(gradient)
            hidden_layer.W = hidden_layer.W - gradient_W * self.lr
            hidden_layer.b = hidden_layer.b - gradient_b * self.lr

        gradient = self.activations[0].gradients(gradient)
        gradient, gradient_W, gradient_b = self.input_layer.gradients(gradient)
        self.input_layer.W = self.input_layer.W - gradient_W * self.lr
        self.input_layer.b = self.input_layer.b - gradient_b * self.lr


class Dense():
    """Dense layer class.

    Takes the form

        dense(x) = W*x + b

    Attributes:
        input_units: Number of input units.
        output_units: Number of output units.
    """

    def __init__(self, input_units, output_units):
        self.W = np.random.random((input_units, output_units))
        self.b = np.zeros(output_units)
        self.x = None

    def forward(self, x):
        self.x = x
        return self.x @ self.W + self.b

    def gradients(self, grad):
        """
        Args:
            grad: Gradient of loss with respect to dense layer.

        Returns:
            Tuple of size 3, containing (gradient with respect to input,
            gradient with respect to weight matrix W, gradient with respect to
            bias matrix b)
        """
        grad_input = grad @ self.W.T
        grad_W = self.x.T @ grad
        grad_b = grad.sum(axis=0)
        assert(grad_W.shape == self.W.shape)
        assert(grad_b.shape == self.b.shape)
        return (grad_input, grad_W, grad_b)


class ReLU():
    """Rectified linear unit class.

    Takes the form

        relu(x) = max(x, 0)

    The gradient of this unit with respect to x is

        _drelu(x)_ = 1 if x > 0, else 0
            dx
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(self.x, 0)

    def gradients(self, grad):
        """
        Args:
            grad: Gradient of loss with respect to ReLU.

        Returns:
            Gradient of loss with respect to input.
        """
        relu_grad = self.x > 0
        return grad * relu_grad

