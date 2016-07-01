import numpy as np

#############################################################################
### INITIALIZATION METHODS                                                ###
#############################################################################


def initialize_network_uniform(n, k, intermediate_sizes):
    dimensions = [n] + intermediate_sizes + [k] # input/output sizes
    weights = [0] * (len(dimensions)-1) # the neurons themselves
    biases = [0] * (len(dimensions)-1)
    
    for i in range(0, len(weights)):
        weights[i] = np.random.random((dimensions[i], dimensions[i+1]))*2 - 1
        biases[i] = np.random.random((1, dimensions[i+1]))*2 - 1
    
    return weights, biases


def initialize_xavier_sigmoid(n, k, intermediate_sizes):
    dimensions = [n] + intermediate_sizes + [k] # input/output sizes
    weights = [0] * (len(dimensions)-1) # the neurons themselves
    biases = [0] * (len(dimensions)-1)
    
    for i in range(0, len(weights)-1):
        r = 4 * ((6/(dimensions[i]+dimensions[i+1]))**0.5)
        weights[i] = np.random.random((dimensions[i], dimensions[i+1]))*(2*r) - r
        biases[i] = np.zeros((1, dimensions[i+1]))
    
    # set the last ones to zero
    weights[-1] = np.zeros((dimensions[-2], dimensions[-1]))
    biases[-1] = np.zeros((1, dimensions[-1]))
    
    return weights, biases

def initialize_xavier_tanh(n, k, intermediate_sizes):
    dimensions = [n] + intermediate_sizes + [k] # input/output sizes
    weights = [0] * (len(dimensions)-1) # the neurons themselves
    biases = [0] * (len(dimensions)-1)
    
    for i in range(0, len(weights)-1):
        r = ((6/(dimensions[i]+dimensions[i+1]))**0.5)
        weights[i] = np.random.random((dimensions[i], dimensions[i+1]))*(2*r) - r
        biases[i] = np.zeros((1, dimensions[i+1]))
    
    # set the last ones to zero
    weights[-1] = np.zeros((dimensions[-2], dimensions[-1]))
    biases[-1] = np.zeros((1, dimensions[-1]))
    
    return weights, biases

def initialize_xavier_leru(n, k, intermediate_sizes):
    dimensions = [n] + intermediate_sizes + [k] # input/output sizes
    weights = [0] * (len(dimensions)-1) # the neurons themselves
    biases = [0] * (len(dimensions)-1)
    
    for i in range(0, len(weights)-1):
        r = (2/(dimensions[i]))**0.5
        weights[i] = np.random.standard_normal((dimensions[i], dimensions[i+1]))*r
        biases[i] = np.zeros((1, dimensions[i+1]))
    
    # set the last ones to zero
    weights[-1] = np.zeros((dimensions[-2], dimensions[-1]))
    biases[-1] = np.zeros((1, dimensions[-1]))
    
    return weights, biases



#############################################################################
### PREDICTION METHODS                                                    ###
#############################################################################

def forward_prop(weights, biases, activations, input_data):
    l = len(weights)
    
    x = [0] * l # input to level i
    z = [0] * l # un-activated output of level i
    y = [0] * l # activated output of level i
    
    x[0] = input_data
    
    for i in range(0, l):
        expanded_bias = np.ones((x[i].shape[0], 1)) * biases[i]
        z[i] = np.dot(x[i], weights[i]) + expanded_bias
        y[i] = activations[i](z[i])
        
        if i < l-1:
            x[i+1] = y[i]
    
    return x, z, y

def forward_prop_dropout(weights, biases, activations, input_data, drop_rates):
    
    l = len(weights)
    
    x = [0] * l # input to level i
    z = [0] * l # un-activated output of level i
    y = [0] * l # activated output of level i
    mask = [0] * l # dropout mask for the input to level i
    
    # Apply a dropout mask and scaling to x[0]
    mask[0] = np.random.random(input_data.shape) > drop_rates[0]
    x[0] = input_data * (mask[0]) / (1-drop_rates[0])
    
    for i in range(0, l):
        expanded_bias = np.ones((x[i].shape[0], 1)) * biases[i]
        z[i] = np.dot(x[i], weights[i]) + expanded_bias
        y[i] = activations[i](z[i])
        
        if i < l-1:
            mask[i+1] = np.random.random(y[i].shape) > drop_rates[i+1]
            x[i+1] = y[i] * (mask[i+1]) / (1-drop_rates[i+1])
    
    return x, z, y, mask

#############################################################################
### ACTIVATION FUNCTIONS                                                  ###
#############################################################################


def act_sigmoid(z, y=None, diff=False):
    if diff:
        if y is None:
            ex = np.exp(-z)
            return -ex / ((1+ex)**2)
        else:
            return y*(1-y)
    else:
        return 1 / (1 + np.exp(-z))

def act_tanh(z, y=None, diff=False):
    if diff:
        if y is None:
            y = np.tanh(z)
        return 1-y*y
    else:
        return np.tanh(z)

def act_identity(z, y=None, diff=False):
    if diff:
        return 1
    else:
        return z

def act_LeRU_maker(leakage=0):
    def f(z, y=None, diff=False):
        if diff:
            return np.where(z>0, 1, leakage)
        else:
            return np.where(z>0, z, leakage*z)
    
    return f

act_LeRU = act_LeRU_maker(leakage=0)

def act_softplus(z, y=None, diff=False):
    if diff:
        return 1 / (1 + np.exp(-z))
    else:
        return np.log(1+np.exp(z))



#############################################################################
### COST FUNCTIONS                                                        ###
#############################################################################


def cost_MSE(y_hat, y, diff=False, aggregate=False):
    if diff:
        return 2 * (y_hat-y)
    elif aggregate:
        # sum-square each row, then take the mean across all rows
        return np.mean(np.sum((y-y_hat)**2, axis=1), axis=0)
    else:
        # sum-square each row
        return np.sum((y-y_hat)**2, axis=1)


def cost_CE(y_hat, y, diff=False, aggregate=False):
    # Prevents divide by zero problems
    y_hat = np.clip(y_hat, np.exp(-36), 1-np.exp(-36))
    
    # Assumes y consists entirely of zeros and ones!
    if diff:
        deriv = np.where(y == 0, 1/(1-y_hat), (-1)/y_hat)
        return deriv
    elif aggregate:
        cost = np.where(y == 0, -np.log(1-y_hat), -np.log(y_hat))
        # sum the errors for each row, then take the mean across all rows
        return np.mean(np.sum(cost, axis=1), axis=0)
    else:
        cost = np.where(y == 0, -np.log(1-y_hat), -np.log(y_hat))
        return cost
        


#############################################################################
### REGULARIZATION METHODS                                                ###
#############################################################################

def ridge_cost(l2_lambda, weights, biases,
               diff=False, aggregate=False,
               regularize_bias=False):
    
    L = len(weights)
    
    if diff:
        scale = 2*l2_lambda
        weight_grad = [scale*weights[i] for i in range(0, L)]
        
        if regularize_bias:
            biases_grad = [scale*biases[i] for i in range(0, L)]
            return weight_grad, biases_grad
        
        else:
            return weight_grad
    
    elif aggregate:
        cost = l2_lambda * sum([np.sum(weights[i]**2) for i in range(0, L)])
        if regularize_bias:
            cost += l2_lambda * sum([np.sum(biases[i]**2) for i in range(0, L)])
            
        return cost
    
    else:
        weight_cost = [l2_lambda * (weights[i]**2) for i in range(0, L)]
        
        if regularize_bias:
            biases_cost = [np.zeros(biases[i].shape) for i in range(0, L)]
            return weight_cost, biases_cost
        
        else:
            return weight_cost

def lasso_cost(l1_lambda, weights, biases,
               diff=False, aggregate=False,
               regularize_bias=False):
    L = len(weights)
    
    if diff:
        weight_grad = [l1_lambda*np.sign(weights[i]) for i in range(0, L)]
        
        if regularize_bias:
            biases_grad = [l1_lambda*np.sign(biases[i]) for i in range(0, L)]
            return weight_grad, biases_grad
        else:
            return weight_grad
    
    elif aggregate:
        cost = l1_lambda * sum([np.sum(np.abs(weights[i])) for i in range(0, L)])
        
        if regularize_bias:
            cost += l1_lambda * sum([np.sum(np.abs(biases[i])) for i in range(0, L)])
            
        return cost
    
    else:
        weight_cost = [l1_lambda * np.abs(weights[i]) for i in range(0, L)]
        
        if regularize_bias:
            biases_cost = [l1_lambda * np.abs(biases[i]) for i in range(0, L)]
            return weight_cost, biases_cost
        else:
            return weight_cost




#############################################################################
### SCORING METHODS                                                       ###
#############################################################################

def classification_success_rate(y_hat, y):
    predicted_classes = np.argmax(y_hat, axis=1)
    actual_classes = np.argmax(y, axis=1)
    errors = predicted_classes - actual_classes
    
    return 1 - (np.count_nonzero(errors) / len(errors))



#############################################################################
### BATCHING METHODS                                                      ###
#############################################################################

def get_mini_batches(batch_size, X, Y):
    scrambled_indices = np.random.permutation(len(X))
    
    batch_edges = list(range(0, len(X), batch_size)) + [len(X)]
    num_batches = len(batch_edges)-1
    
    for i in range(0, num_batches):
        batch_indices = scrambled_indices[batch_edges[i]:batch_edges[i+1]]
        yield X[batch_indices], Y[batch_indices]

#############################################################################
### TRAINING METHODS                                                      ###
#############################################################################


def back_prop(weights, biases, acts, cost_function,
              train_X, train_Y,
              x, y, z):
    L = len(weights) # number of layers
    
    cost_diff = cost_function(y[-1], train_Y, diff=True)
    
    # Gradient of cost at each level
    bp_grad = [0] * L
    
    # The last level is special
    bp_grad[L-1] = cost_diff * acts[L-1](z[L-1], y[L-1], diff=True)
    
    # The rest of the levels are just gotten by propagating backward
    for i in range(L-2, -1, -1):
        scaled_grad = bp_grad[i+1] * acts[i+1](z[i+1], y[i+1], diff=True)
        bp_grad[i] = np.dot(scaled_grad, weights[i+1].T)
    
    # Now adjust for the weights and biases themselves
    bp_grad_w = [0] * L
    bp_grad_b = [0] * L
    
    for i in range(0, L):
        scaled_grad = bp_grad[i] * acts[i](z[i], y[i], diff=True)

        bp_grad_w[i] = np.dot(x[i].T, scaled_grad)
        
        relevant_ones = np.ones((1, x[i].shape[0]))
        bp_grad_b[i] = np.dot(relevant_ones, scaled_grad)
        
    return bp_grad_w, bp_grad_b

    def back_prop_dropout(weights, biases, acts, cost_function,
                          train_X, train_Y,
                          x, y, z, masks):
    L = len(weights) # number of layers
    
    cost_diff = cost_function(y[-1], train_Y, diff=True)
    
    # Gradient of cost at each level
    bp_grad = [0] * L
    
    # The last level is special
    bp_grad[L-1] = cost_diff * acts[L-1](z[L-1], y[L-1], diff=True)
    
    # The rest of the levels are just gotten by propagating backward
    for i in range(L-2, -1, -1):
        scaled_grad = bp_grad[i+1] * acts[i+1](z[i+1], y[i+1], diff=True)
        bp_grad[i] = np.dot(scaled_grad, weights[i+1].T)
    
    # Now adjust for the weights and biases themselves
    bp_grad_w = [0] * L
    bp_grad_b = [0] * L
    
    for i in range(0, L):
        if i < L-1:
            scaled_grad = bp_grad[i] * acts[i](z[i], y[i], diff=True) * masks[i+1]
        else:
            scaled_grad = bp_grad[i] * acts[i](z[i], y[i], diff=True)

        bp_grad_w[i] = np.dot(x[i].T, scaled_grad)
        
        relevant_ones = np.ones((1, x[i].shape[0]))
        bp_grad_b[i] = np.dot(relevant_ones, scaled_grad)
        
    return bp_grad_w, bp_grad_b