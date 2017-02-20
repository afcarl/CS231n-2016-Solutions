import numpy as np

def rel_error(x, y):
    """Relative error
    """
    return np.max(np.abs(x - y)/np.maximum(1e-8, np.abs(x) + np.abs(y)))

def eval_numerical_gradient(f, x, verbose=False, h=1e-5):
    """Numerical gradient of function f at x
    
    Args:
        f: A function that takes a single argument
        x: A numpy array
        verbose: A bool, True: print out details
        h: A float, step size
        
    Returns:
        grad: A numpy array, numerical gradient
    """
    
    # Initialize grad
    grad = np.zeros_like(x)
    
    # Evaluate function f at x
    y = f(x)
    
    # Iterate all indices of x
    iter_x = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not iter_x.finished:
        idx = iter_x.multi_index
        
        oldval = x[idx]
        x[idx] = oldval + h # Increment by h
        f_xplush = f(x) # Evaluate f(x + h)
        x[idx] = oldval - h # Decrement by h
        f_xminush = f(x) # Evaluate f(x - h)
        x[idx] = oldval # Restore
        
        # Compute partial derivative with centered formula
        grad[idx] = (f_xplush - f_xminush)/(2*h)
        if verbose:
            print(idx, grad[idx])
        iter_x.iternext()
        
    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """Evaluate numerical gradient
    
    Args:
        f: A function of x
        x: A numpy array
        df: 
        h: A float, step size
        
    Returns:
        grad: A numpy array, numerical gradient
    """
    
    # Initialize grad
    grad = np.zeros_like(x)
    
    # Iterate all indices of x
    iter_x = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not iter_x.finished:
        idx = iter_x.multi_index
        
        oldval = x[idx]
        x[idx] = oldval + h # Increment by h
        f_xplush = f(x).copy() # Evaluate f(x + h)
        x[idx] = oldval - h # Decrement by h
        f_xminush = f(x).copy() # Evaluate f(x - h)
        x[idx] = oldval # Restore
        
        # Compute partial derivative with centered formula
        grad[idx] = np.sum((f_xplush - f_xminush)*df)/(2*h)
        iter_x.iternext()
        
    return grad

