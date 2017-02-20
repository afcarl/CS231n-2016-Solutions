import numpy as np

def SGD(w, dw, config=None):
    """Vanilla stochastic gradient descent.
    
    Args:
        w: A numpy array, current weights
        dw: A numpy array, same shape with w, the gradient of the loss w.r.t w.
        config: A dictionary
            learning_rate: A float
    
    Returns:
        next_w: A numpy array, same shape with w, updated weight
        config: A dictionary, passing to next update iteration
    """
    
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    
    s = -config['learning_rate']*dw
    next_w = w + s
    
    return next_w, config

def SGDMomentum(w, dw, config=None):
    """Stochastic gradient descent with momentum.
    
    Args:
        w: A numpy array, current weights
        dw: A numpy array, same shape with w, the gradient of the loss w.r.t w.
        config: A dictionary
            learning_rate: A float
            momentum: A float in [0, 1]. Setting 0 results in SGD
            velocity: A numpy array, same shape with w and dw, a moving average of the gradients
    
    Returns:
        next_w: A numpy array, same shape with w, updated weight
        config: A dictionary, passing to next update iteration
    """
    
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w)) # get() with default return when key is absent, no exception
    
    # Update (integrate) velocity
    v = config['momentum']*v - config['learning_rate']*dw
    # Parameter update (integrate position)
    next_w = w + v
    
    config['velocity'] = v
    
    return next_w, config

# In following, there are per-parameter adaptive learning rate methods

def Adagrad(w, dw, config=None):
    """Adaptive subgradient update rule.
    It keeps tracking of the sum of squared gradients, and use it to normalize
    the update step element-wise(per-paramter). 
        Weights with large gradients -> reduced effective learning rate
        Weights with small gradients -> increase effective learning rate
    
    Args:
        w: A numpy array, current weights
        dw: A numpy array, same shape with w, the gradient of the loss w.r.t w.
        config: A dictionary
            learning_rate: A float
            epsilon: A float, small value for avoiding zero division
            cache: A numpy array, same shape with w and dw, 
                   moving average of the second moment gradients.
    
    Returns:
        next_w: A numpy array, same shape with w, updated weight
        config: A dictionary, passing to next update iteration
    """
    
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
    
    config['cache'] += dw**2
    next_w = w - config['learning_rate']*dw/(np.sqrt(config['cache']) + config['epsilon'])
    
    return next_w, config

def RMSprop(w, dw ,config=None):
    """RMSprop: using a moving average of squared gradient values
    to set adaptive per-parameter learning rates.
    
    Args:
        w: A numpy array, current weights
        dw: A numpy array, same shape with w, the gradient of the loss w.r.t w.
        config: A dictionary
            learning_rate: A float
            decay_rate: A float in [0, 1], the decay rate for the squared gradient
            epsilon: A float, small value for avoiding zero division
            cache: A numpy array, same shape with w and dw, 
                moving average of the second moment gradients.
    
    Returns:
        next_w: A numpy array, same shape with w, updated weight
        config: A dictionary, passing to next update iteration
    """
    
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
    
    config['cache'] = config['decay_rate']*config['cache'] + (1 - config['decay_rate'])*(dw**2)
    next_w = w - config['learning_rate']*dw/(np.sqrt(config['cache']) + config['epsilon'])
    
    return next_w, config

def Adam(w, dw, config=None):
    """Adam: incorperates moving averages of both gradient, its square, and a biase correction
    
    Args:
        w: A numpy array, current weights
        dw: A numpy array, same shape with w, the gradient of the loss w.r.t w.
        config: A dictionary
            learning_rate: A float
            beta1: Decay rate for moving average of first moment of gradient
            beta2: Decay rate for moving average of second moment of gradient
            m: A numpy array, same shape with w, moving average of gradient
            v: A numpyp array, same shape with w, moving average of squared gradient
            t: Iteration number
            epsilon: A float, small value for avoiding zero division
    
    Returns:
        next_w: A numpy array, same shape with w, updated weight
        config: A dictionary, passing to next update iteration
    """
    
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('epsilon', 1e-8)
    config.setdefault('t', 0)
    
    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    
    config['t'] += 1 # Increment and avoid 0-th iteration
    
    config['m'] = beta1*config['m'] + (1 - beta1)*dw # update first moment
    config['v'] = beta2*config['v'] + (1 - beta2)*(dw**2) # update second moment
    # Correct bias
    mb = config['m']/(1 - beta1**config['t'])
    vb = config['v']/(1 - beta2**config['t'])
    # Update
    next_w = w - learning_rate*mb/(np.sqrt(vb) + config['epsilon'])
    
    return next_w, config