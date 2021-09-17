import functions as f

#implement two-layer-model
def two_layer_model(X, Y, layer_dims, learning_rate=0.001, num_iterations=3000, print_cost=False):
    np.random.seed(123)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims

    parameters = f.initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        #LINEAR -> RELU -> LINEAR -> SIGMOID
        A1, cache1 = f.linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = f.linear_activation_forward(A1, W2, b2, 'sigmoid')

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = f.linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = f.linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = f.update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    return parameters

#implement L-layer model
def L_layer_model(X, Y, layer_dims, learning_ratae=0.001, num_iterations=3000, print_cost=False):
    np.random.seed(123)
    costs = []

    parameters = f.initialize_parameters_many(layer_dims)

    for i in range(0, num_iterations):
        AL, caches = f.L_model_forward(X, parameters)

        cost  = f.compute_cost(AL, Y)

        grads = f.L_model(backward(AL, Y, caches))

        parameters = f.update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters
