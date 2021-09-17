import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s

def relu(z):
    r = np.maximum(0, z)

    return r

def sigmoid_backward(dA, cache):

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ

def relu_backward(dA, cache):

    Z = cache

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ

#parameter initialize
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(123)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

#parameter initialize_many_vrsion
def initialize_parameters_many(layer_dims):

    np.random.seed(123)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

#forward propagation
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache

#forward propagation [LINEAR -> ACTIVATION Layer]
def linear_activation_forward(A_prev, W, b, activation):

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

#forward propagation [ABOVE MODEL]
def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches

#cost function
def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = -1 / m * np.sum(np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T))

    cost = np.squeeze(cost)

    return cost

#linear portion of backward propagation for a single layer
def linear_backward(dZ, cache):

    A_prev, W, b= cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

#backward propagation [LINEAR->ACTIVATION Layer]
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

#backward propagation [LINEAR->RELU] * (L-1) ->LINEAR -> SIGMOID Model
def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads['dA'+str(L-1)] = dA_prev_temp
    grads['dW'+str(L)] = dW_temp
    grads['db' + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+1)], current_cache, 'relu')
        grads['dA'+str(l)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp

    return grads

#update parameters
def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

    return parameters


