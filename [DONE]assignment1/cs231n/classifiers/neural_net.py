import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    
    self.network = ["Input", "FC1", "ReLU", "FC2", "Softmax", "Output_loss"]

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    '''
    layer1 = X.dot(W1) + b1
    layer1_relu = np.maximum(0, layer1) # ReLU
    layer2 = layer1_relu.dot(W2) + b2
    
    scores = layer2
    '''
    
    if y is None:
        self.network[-2] = "Softmax_score"
    else:
        self.network[-2] = "Softmax_loss"
        
    forward_results = self.forward(X, y)
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      #return scores
      return forward_results["FC2"]

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    '''
    scores -= np.row_stack(np.max(scores, 1)) # solve numeric instability
    scores = np.exp(scores)
    scores /= np.row_stack(np.sum(scores, 1)).astype(float)
    
    loss = np.sum(-np.log(scores[xrange(N), y]))/float(N) + 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
    '''
    loss = forward_results["Softmax_loss"]
    loss += 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    '''
    # Backprop in softmax
    dW2_term = scores; dW2_term[xrange(N), y] -= 1; dW2_term /= float(N)
    grads["W2"] = (layer1_relu.T).dot(dW2_term) + reg*W2
    grads["b2"] = np.sum(dW2_term, 0)
    grads["W1"] = 
    '''
    
    backward_results = self.backward(X, y, forward_results, reg)
    
    grads["W1"] = backward_results["FC1"]["grad_W"]
    grads["b1"] = backward_results["FC1"]["grad_b"]
    grads["W2"] = backward_results["FC2"]["grad_W"]
    grads["b2"] = backward_results["FC2"]["grad_b"]
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_ind = np.random.choice(num_train, batch_size)
      X_batch = X[batch_ind]
      y_batch = y[batch_ind]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params["W1"] += -learning_rate*grads["W1"]
      self.params["b1"] += -learning_rate*grads["b1"]
      self.params["W2"] += -learning_rate*grads["W2"]
      self.params["b2"] += -learning_rate*grads["b2"]
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(self.loss(X), 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

  def forward_step(self, Input, y, layer_type):
        layer_type = layer_type.lower() # avoid case sensitive for comparison
        if layer_type == "Input".lower():
            return Input
        elif layer_type == "Output_loss".lower():
            return Input
        elif layer_type.startswith("FC".lower()):
            return Input.dot(self.params["W" + layer_type[-1]]) + self.params["b" + layer_type[-1]]
        elif layer_type.startswith("ReLU".lower()):
            return np.maximum(0, Input)
        elif layer_type == "Softmax_score".lower():
            tmp = Input - np.row_stack(np.max(Input, 1)) # solve numeric instability
            return np.exp(tmp)/np.row_stack(np.sum(tmp, 1)).astype(float)
        elif layer_type == "Softmax_loss".lower():
            tmp = Input - np.row_stack(np.max(Input, 1)) # solve numeric instability
            tmp = np.exp(tmp)
            tmp = tmp/np.row_stack(np.sum(tmp, 1)).astype(float)
            
            return np.sum(-np.log(tmp[xrange(tmp.shape[0]), y]))/float(Input.shape[0])
            
  def forward(self, X, y):
    result = X
    results = {}
    for layer_type in self.network:
        result = self.forward_step(result, y, layer_type)
        results[layer_type] = result
    return results

  def backward_step(self, y, forward_data, previous_grad, layer_type, reg):
        # Take out forward data
        step_outdata = forward_data[layer_type]
        step_indata = forward_data[self.network[self.network.index(layer_type) - 1]] if layer_type != "Input".lower() else None
        
        layer_type = layer_type.lower() # avoid case sensitive for comparison
        if layer_type == "Output_loss".lower():
            return 1
        elif layer_type == "Input".lower():
            return previous_grad # Outer API to parser with key ["grad_data"]
        elif layer_type.startswith("FC".lower()):
            param_W = self.params["W" + layer_type[-1]]
            param_b = self.params["b" + layer_type[-1]]
            result = {}
            result["grad_data"] = previous_grad.dot(param_W.T)
            result["grad_W"] = (step_indata.T).dot(previous_grad)
            if reg != 0: result["grad_W"] += reg*param_W
            result["grad_b"] = np.sum(previous_grad, 0)
            return result
        elif layer_type == "Softmax_loss".lower():
            tmp = step_indata - np.row_stack(np.max(step_indata, 1)) # solve numeric instability
            tmp = np.exp(tmp)
            tmp /= np.row_stack(np.sum(tmp, 1)).astype(float)
            tmp[xrange(tmp.shape[0]), y] -= 1
            return tmp/float(tmp.shape[0])
        elif layer_type == "ReLU".lower():
            previous_grad[step_indata <= 0] = 0
            return previous_grad
            
            
  def backward(self, X, y, forward_data, reg = 0):
    results = {}
    
    previous_grad = 1
    for layer_type in self.network[::-1]: # backward loop
        if type(previous_grad) == dict: previous_grad = previous_grad["grad_data"]
        results[layer_type] = previous_grad = self.backward_step(y, forward_data, previous_grad, layer_type, reg)
    return results