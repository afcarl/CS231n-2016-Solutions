import numpy as np
from utils import optimizers

class Solver(object):
    """A Solver accepts training and validation data with labels for training. 
    It will periodically check training and validation accuracies to monitor overfitting.  
    
    To train a model: construct Solver instance with model, dataset and options(batch size etc.)
        then call train() method.
        
    After train() call, model.params contains best parameters on validation set during training.
    solver.loss_history contains list of all losses during training
    solver.train_acc_history contains list of all accuraries on training set at each epoch.
    solver.val_acc_history contains list of all accuraries on validation set at each epoch.
    
    A Solver works on a model object that must conform the following API:
        model.params: A dictionary mapping string names to numpy array
        model.loss(X, y): A function to compute training loss, gradients and test-time scores that
            Args:
                X: A numpy array, shape [N, d_1, ..., d_k], minibatch of data
                y: A numpy array, shape [N, ], labels where y[i] is the label for X[i]
            
            Returns:
                If y is None: test-time forward pass and return
                    scores: A numpy array, shape [N, C]
                If y is not None: training-time forward pass and return
                    loss: A float
                    grads: A dictionary, keys are string names, values are gradients
    """
    
    def __init__(self, model, data, **kwargs):
        """Construct a Solver instance
        
        Required args:
            model: A model object conforming to API described above
            data: A dictionary of training and validation data such that
                'X_train': A numpy array, shape [N_train, d_1, ..., d_k]
                'y_train': A numpy array, shape [N_train, ]
                'X_val': A numpy array, shape [N_val, d_1, ..., d_k]
                'y_val': A numpy array, shape [N_val, ]
        
        Optional args:
            update_rule: A string, the name of update rule in optimizers.py
            optim_config: A dictionary, hyperparameters for update rule.
            lr_decay: A float, learning rate decay. After each epoch, the learning rate
                is multiplied by this value
            batch_size: An int, the minibatch size to compute loss and gradient during training
            num_epochs: An int, the number of epochs
                i.e. 1 epoch = 1 round of all training set = [N_train/batch_size] iterations
            print_every: An int, print training loss every print_every iterations
            verbose: A bool, False: no output to print during training
        """
        
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        
        # Unpack keyword args
        self.update_rule = kwargs.pop('update_rule', 'SGD')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        
        # Throw an exception if there are extra keyword args
        if len(kwargs) > 0:
            extra = ', '.join('{}'.format(k) for k in kwargs.keys())
            raise ValueError('Unrecognized arguments: {}'.format(extra))
            
        # Ensure the update rule exists in optimizers.py
        if not hasattr(optimizers, self.update_rule):
            raise ValueError('Invalid update_rule {}'.format(self.update_rule))
        # Obtain update_rule as a function
        self.update_rule = getattr(optimizers, self.update_rule)
        
        self._reset()
        
    def _reset(self):
        """Helper function: set up book-keeping variables for optimization
        """
        # Set up some variables
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        # Make a deep copy of optim_config for each parameter
        self.optim_configs = {}
        for param in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[param] = d
            
    def _step(self):
        """Helper function: A single gradient update. Called by train()
        """
        
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
        
        # Parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
            
    def check_accuracy(self, X, y, num_samples=None, batch_size=128):
        """Check accuracy of the model
        
        Args:
            X: A numpy array, shape [N, d_1, ..., d_k], input data
            y: A numpy array, shape [N, ], labels
            num_samples: An int
                If not None: subsample the data and test model on num_samples datapoints
            batch_size: An int
            
        Returns:
            acc: A float, the fraction of correct instances
        """
        
        N = X.shape[0]
        
        # Subsample the data if required
        if num_samples is not None and num_samples < N:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
            
        # Compute predictions in batches
        num_batches = N//batch_size
        y_pred = []
        # For residual datapoints in final batch
        if N % batch_size != 0:
            num_batches += 1
        for i in range(num_batches):
            # Start and end indices for a batch 
            start = i*batch_size
            end = (i + 1)*batch_size
            
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred) # make a single vector
        acc = np.mean(y_pred == y)
        
        return acc
    
    def train(self):
        """Train the model
        """
        
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(1, num_train//self.batch_size)
        num_iterations = self.num_epochs*iterations_per_epoch
        
        for t in range(num_iterations):
            # Make a single update step
            self._step()
            
            # Print training loss if required
            if self.verbose and t % self.print_every == 0:
                print('(Iteration {} / {}) loss: {}'.format(t + 1, 
                                                            num_iterations, 
                                                            self.loss_history[-1]))
            
            # At the end of each epoch, increment epoch counter and decay learning rate
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
                    
            # Check train and val accuracy on first and last iteration, and 
            # at the end of each epoch
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                
                if self.verbose:
                    print('Epoch {} / {} train acc: {}; val_acc: {}'.format(self.epoch,
                                                                            self.num_epochs,
                                                                            train_acc, 
                                                                            val_acc))
                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
            
        # At the end of training, swap the best params into the model
        self.model.params = self.best_params