import numpy as np

def affine_forward(x, w, b):
    """Compute the forward pass for an affine (Fully-connected) layer.
    
    Args:
        x: A numpy array, shape [N, d_1, ..., d_k], input data. 
            Need to convert it to shape [N, D] where D = d_1*...*d_k
        w: A numpy array, shape [D, M], weights
        b: A numpy array, shape [M, ], biases
        
    Returns:
        out: A numpy array, shape [N, M], output
        cache: A tuple, (x, w, b)
    """
    
    # Convert input array to shape [N, D]
    x2vec = x.reshape(x.shape[0], -1)
    
    # Computation of affine layer
    out = x2vec.dot(w) + np.column_stack(b)
    
    cache = (x, w, b)
    
    return out, cache

def affine_backward(dout, cache):
    """Computes the backward pass for an affine layer.
    
    Args:
        dout: Upstream derivative, shape [N, M]
        cache: cache used in forward pass, (x, w, b)
        
    Returns:
        dx: A numpy array, shape [N, d_1, ..., d_k], gradient w.r.t. x
        dw: A numpy array, shape [D, M], gradient w.r.t. w
        db: A numpy array, shape [M, ], gradient w.r.t. b
    """
    
    x, w, b = cache
    
    # Convert x to shape [N, D]
    x_reshaped = x.reshape([x.shape[0], -1])
    
    # Compute gradients
    dx = dout.dot(w.T).reshape(x.shape)
    dw = (x_reshaped.T).dot(dout)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db

def relu_forward(x):
    """Compute the forward pass for a ReLU layer
    
    Args:
        x: A numpy array, any shape, input data
        
    Returns:
        out: A numpy array, same shape as x, output
        cache: x
    """
    
    out = np.maximum(x, 0)
    cache = x
    
    return out, cache
    
def relu_backward(dout, cache):
    """Compute the backward pass for ReLU layer
    
    Args:
        dout: Upstream derivatives, any shape
        cache: Input data x, same shape as dout
    Returns:
        dx: Gradient w.r.t. x
    """
    
    x = cache
    
    x[x <= 0] = 0
    x[x > 0] = 1
    dx = x*dout
    
    return dx

def sigmoid_forward(x):
    """Compute forward pass for Sigmoid layer
    
    Args:
        x: A numpy array, any shape, input data
        
    Returns:
        out: A numpy array, same shape as x, output
        cache: out
    """
    
    out = 1/(1 + np.exp(-x))
    cache = out
    
    return out, cache

def sigmoid_backward(dout, cache):
    """Compute the backward pass for Sigmoid layer
    
    Args:
        dout: Upstream derivatives, any shape
        cache: Input data x, same shape as dout
    Returns:
        dx: Gradient w.r.t. x
    """
    
    out = cache
    
    dx = out*(1 - out)
    dx = dout*dx
    
    return dx

def tanh_forward(x):
    """Compute the forward pass for tanh layer
    
    Args:
        x: A numpy array, any shape, input data
        
    Returns:
        out: A numpy array, same shape as x, output
        cache: out
    """
    
    out = np.tanh(x)
    cache = out
    
    return out, cache

def tanh_backward(dout, cache):
    """Compute the backward pass for tanh layer
    
    Args:
        dout: Upstream derivatives, any shape
        cache: Input data x, same shape as dout
    Returns:
        dx: Gradient w.r.t. x
    """
    
    out = cache
    dx = 1 - out**2
    dx = dout*dx
    
    return dx

def dropout_forward(x, dropout_param):
    """Forward pass for (inverted) dropout
    i.e. divided by p during training, no change on test time.
    
    Args:
        x: A numpy array, any shape
        dropout_param: A dictionary
            p: A float in [0, 1], drop each neuron output with probability p.
            mode: 'train'/'test'. If 'test': just return input without dropout
            seed: For random number generator. To be deterministic, for gradient check.
    
    Returns:
        out: A numpy array, same shape with x
        cache: A tuple (dropout_param, mask)
                In train mode: mask is dropout mask
                In test mode: mask is None
    """
    
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
        
    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p)/p # divided by p: then no need to change test time code
        out = mask*x
    elif mode == 'test':
        mask = None
        out = x
    
    out = out.astype(x.dtype, copy=False)
    cache = (dropout_param, mask)
    
    return out, cache

def dropout_backward(dout, cache):
    """Backward pass for (inverted) dropout.
    i.e. divided by p during training, no change on test time.
    
    Args:
        dout: A numpy array, any shape, upstream derivatives
        cache: A tuple (dropout_param, mask) from forward pass
        
    Returns:
        dx: A numpy array, same shape with x, gradients w.r.t x
    """
    
    dropout_param, mask = cache
    mode = dropout_param['mode']
    
    if mode == 'train':
        dx = dout*mask
    elif mode == 'test':
        dx = dout
        
    return dx

def softmax_loss(x, y):
    """Compute loss and gradient of softmax
    
    Args:
        x: A numpy array, shape [N, C], logits
        y: A numpy array, shape [N,], training labels, i.e. y[i] = c -> X[i] has label c
        
    Returns:
        loss: A float
        dx: A numpy array, same shape with x, gradient of loss w.r.t x. 
    """
    
    # Get the number of training examples N
    N = y.size
    
    # Compute probabilities with solving numeric instability
    prob = np.exp(x - np.max(x, axis=1, keepdims=True)) # keepdims=True correctly broadcast array
    prob /= np.sum(prob, axis=1, keepdims=True)
    
    # Compute loss
    loss = np.sum(-np.log(prob[range(N), y]))/N
    
    # Compute gradient
    dx = prob.copy()
    dx[range(N), y] -= 1
    dx /= N
    
    return loss, dx