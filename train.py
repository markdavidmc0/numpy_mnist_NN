import numpy as np
import matplotlib.pyplot as plt
import mnist
import sklearn
import sklearn.datasets
import sklearn.linear_model


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return params


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = W1.dot(np.transpose(X)) + b1
    A1 = np.tanh(Z1)
    Z2 = W2.dot(A1) + b2
    t = np.exp(Z2)
    A2 = t / np.sum(t, axis=0)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache


def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (10, number of examples)
    Y -- "true" labels vector of shape (10, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    [Note that the parameters argument is not used in this function,
    but the auto-grader currently expects this parameter.
    Future version of this notebook will fix both the notebook
    and the auto-grader so that `parameters` is not needed.
    For now, please include `parameters` in the function signature,
    and also when invoking this function.]

    Returns:
    cost -- cross-entropy cost given equation (13)

    """
    m = Y.shape[0]

    cost = - (1/m) * np.sum(np.multiply(Y.transpose(), np.log(A2)))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = Y.shape[0]

    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - np.transpose(Y)
    dW2 = (1/m) * dZ2.dot(np.transpose(A1))
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.transpose(W2).dot(dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * dZ1.dot(X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        'dW2': dW2,
        'db2': db2,
        'dW1': dW1,
        'db1': db1
    }

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (784, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    n_x = np.shape(X)[1]
    n_y = 10

    params = initialize_parameters(n_x, n_h, n_y)  # initialise params

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, params)  # forward prop

        cost = compute_cost(A2, Y)  # cost function
        if i % 10 == 0:
            print(cost)

        grads = backward_propagation(params, cache, X, Y)  # back prop

        params = update_parameters(params, grads, learning_rate=0.1)  # gradient descent

    return params


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = np.argmax(A2, axis=0)

    return predictions


def performance(predictions, Y_test):
    """
    Using trained model predictions, calculate performance metrics against test set

    Arguments:
    predictions -- python dictionary containing model predictions
    Y_test -- input data of size (n_y, m)

    Returns
    performance_metrics -- dict of accuracy, precision and recall
    """
    accuracy = (np.sum(predictions == Y_test) / Y_test.shape[0])
    precision = {}
    recall = {}
    for n in range(0, 10):
        TP = np.sum((predictions == Y_test) & (predictions == n))
        FP = np.sum((predictions != Y_test) & (predictions == n))
        FN = np.sum((Y_test == n) & (predictions != n))
        precision[f'precision_{n}'] = TP / (TP + FP)
        recall[f'recall_{n}'] = TP / (TP + FN)

    performance_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    return performance_metrics


if __name__ == '__main__':
    # data initialisation
    mnist.temporary_dir = lambda: '/Users/markmc/PycharmProjects/street_theft_NN'
    train_images = mnist.train_images()[0:10000]
    train_labels = mnist.train_labels()[0:10000]

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    X_train = train_images.reshape((train_images.shape[0],
                                    train_images.shape[1] * train_images.shape[2]))
    # Y_train = train_labels.reshape((10000, 1))
    Y_train = train_labels

    X_test = test_images.reshape((test_images.shape[0],
                                  test_images.shape[1] * test_images.shape[2]))
    # Y_test = test_labels.reshape((10000, 1))
    Y_test = test_labels

    # one-hot encoding
    Y_train_hot = np.zeros((Y_train.size, 10))
    Y_test_hot = np.zeros((Y_test.size, 10))
    Y_train_hot[np.arange(Y_train.size), Y_train] = 1
    Y_test_hot[np.arange(Y_test.size), Y_test] = 1

    #  train model
    n_h = 8
    num_iterations = 10000
    parameters = nn_model(X_train, Y_train_hot, n_h, num_iterations=5000)

    # make predictions
    predictions = predict(parameters, X_test)

    # accuracy
    performance_metrics = performance(predictions, Y_test)
