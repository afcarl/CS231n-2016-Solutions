from utils.layers import *

def affine_relu_forward(x, w, b):
    """Composed layer: FC -> ReLU
    
    Args:
        x: A numpy array, shape [N, d_1, ..., d_k], input data. 
            Need to convert it to shape [N, D] where D = d_1*...*d_k
        w: A numpy array, shape [D, M], weights
        b: A numpy array, shape [M, ], biases
        
    Returns:
        out: A numpy array, shape [N, M], output from ReLU
        cache: A tuple, (fc_cache, relu_cache)
    """
    
    out_affine, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(out_affine)
    cache = (fc_cache, relu_cache)
    
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass: FC -> ReLU
    
    Args:
        dout: Upstream derivative, shape [N, M]
        cache: (fc_cache, relu_cache)
        
    Returns:
        dx: A numpy array, shape [N, d_1, ..., d_k], gradient w.r.t. x
        dw: A numpy array, shape [D, M], gradient w.r.t. w
        db: A numpy array, shape [M, ], gradient w.r.t. b
    """
    
    fc_cache, relu_cache = cache
    
    dout_relu = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dout_relu, fc_cache)
    
    return dx, dw, db