"""
This module contains the definitions of several classes: Linear, Optimizer,
Sequential, LossMSE and BCELoss

"""


import math
import torch as t

t.set_grad_enabled(False)

##############################################################################


    
class Linear():
    """
    A class used to represent a linear layer, i.e. a layer where the
    transformation x |--> s = Wx + b takes place.
    ...
    
    Attributes
    ----------
    x : torch tensor (nb_samples_per_batch x in_dim)
        outuput of the previous layer
    in_dim : int
        number of inputs
    out_dim : int
        number of outputs
    w : torch tensor (out_dim x in_dim)
        weight matrix
    b : torch tensor (out_dim)
        bias vector
    gradw : torch tensor (out_dim x in_dim)
        the gradient of the loss funcion with respect to w
    gradb : torch tensor (out_dim)
        the gradient of the loss function with respect to b
        
    Methods
    -------
    forward(Input)
        performs forward pass
    
    backward(gradoutput)
        performs backward pass
    
    zero_grad
        sets gradw and gradb to zero
         
    param
        returns w, b, gradw, gradb
        
    gain
        returns the recommended gain value
    
    Raises
    ------
    TypeError
        If in_dim and out_dim are not integers
    
    """
    
    def __init__(self, in_dim : int, out_dim : int, bias = True, Init = True,
                 Xavier = False, gain = 1.0):
        """
        Parameters
        ----------
        in_dim : int
            input dimentions
        out_dim : int
            output dimentions
        bias : bool, optional
            Determines whether to have a bias vector. The default is True.
        Init : bool, optional
            Determines whether to initialize the weights (using standard or 
            Xavier's initialisation) or not. The default is True. If False,
            weights are sampled from the normal distribution N(0, 0.001)
        Xavier : bool, optional
            Determines whether to use Xavier normal initialisation, or the 
            standard initialisation (adopted by PyTorch). The default is False.
        gain : float, optional
            Gain for the Xavier initialisation. The default is 1.

        Raises
        ------
        TypeError
            If in_dim and out_dim are not integers

        """
        
        if not (isinstance(in_dim, int) and isinstance(out_dim, int)):
            raise TypeError("Linear layer's dimentions should be integers")

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.gradw = t.zeros(out_dim, in_dim)
        self.gradb = t.zeros(out_dim)
        
        if Init:
            if Xavier:
                stdv = gain * math.sqrt(2 / (in_dim + out_dim))
                self.w = t.empty(out_dim, in_dim).normal_(0, stdv)
                if bias : 
                    self.b = t.empty(out_dim).normal_(0, stdv)
                else: 
                    self.b = 0
            
            else:
                # Standard PyTorch initialization
                
                lim = 1/math.sqrt(in_dim)
                self.w = t.empty(out_dim, in_dim).uniform_(-lim, lim)
            
                if bias : 
                    self.b = t.empty(out_dim).uniform_(-lim, lim)
                else: 
                    self.b = 0

        else: 
            self.w = t.randn(out_dim,in_dim)*0.001
            if bias : 
                self.b = t.randn(out_dim)*0.001
            else: 
                self.b = 0
            

    def forward(self, Input):
        """
        Performs forward pass and updates x
        
        Parameters
        ----------
        Input : torch tensor (nb_samples_per_batch x in_dim)
            the new value for class attribute x
        
        Returns
        -------
        s : torch tensor (nb_samples_per_batch x out_dim)
            result of the forward pass s = wx+b
        
        """

        self.x = Input.clone()
        s = t.matmul(Input, self.w.t()) + self.b
        
        return s

    def backward(self, dl_ds):
        """
        Performs backward pass. It updates gradw and gradb. Should not be 
        called unless forward method has been called once before.
        
        Parameters
        ----------
        dl_ds : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the loss with respect to s(l)
        
        Returns
        -------
        dl_dx : torch tensor (nb_samples_per_batch x in_dim)
            the gradient of the loss with respect to x(l-1), which
            is the class attribute x
        """
    
        self.gradw.add_(t.matmul(dl_ds.t(), self.x)) # From the course
        self.gradb.add_(dl_ds.sum(0)) # Proven mathematically
        dl_dx = t.matmul(dl_ds, self.w) # From the course
                
        return dl_dx

    def zero_grad(self):
        """
        Sets gradw and gradb to zero
        """
        self.gradw = t.zeros(self.out_dim, self.in_dim)
        self.gradb = t.zeros(self.out_dim)
        
  
    def param(self):
        """
        Returns a list: [w, b, gradw, gradb]
        """

        return [self.w, self.b, self.gradw, self.gradb]
    
    def gain(self):
        """
        Returns the recommended gain value
        """
        return 1.


##############################################################################


class Optimizer():
    """ 
    A class used to represent an optimizer, whose goal is to update the
    model's parameters using an optimization algorithm. We implemented two:
    the classic gradient descent (with optional momentum) and the Adam
    optimizer.
    
    Attributes
    ----------
    eta : float
        scaling of the step for the gradient descent. The default is 1e-1.
        
    momentum : float
        optional momentum for the gradient descent optimizer. The default is 0
     
    Adam : bool
        determines whether to use the Adam optimizer instead of the classic
        gradient descent optimizer. The default is False. If True, then the
        following attributes are also created:
    
    beta_1 : float
        the default is 0.9.
        
    beta_2 : float
        the default is 0.999.
        
    eps : float
        the default is 1e-8.
        
    t : int
        time step. Initialized to 0.
        
    alpha : float
        scaling of the step for the Adam optimizer
        
    m : list
        biased first moment estimate. Each element of the list, corresponding 
        to a linear layer, is itself a list of two elements. The first, is the
        biased first moment estimate over the dL/dw of that layer. The second,
        is the biased first moment estimate over the dL/db of that layer.
    
    v : list
        biased second raw moment estimate. Structured as m.
        
    Methods
    ------
    step(model_params)
        performs the optimization step with the parameters defined in the
        initialization of the optimizer
    
    """
    
    def __init__(self, eta = 1e-1, Adam = False, RMS = False, momentum = 0,
                 beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8, alpha = 1e-3, psi = 1e-4, nu = 0.95):
        
        self.Adam = Adam
        self.RMS = RMS
        self.eta = eta
        self.momentum = momentum
        self.t = 0
        self.m = []
        self.v = []
        self.phi = []
        
        if Adam :
            self.alpha = alpha
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.eps = eps
            
        if RMS : 
            self.psi = psi
            self.nu = nu
            self.beta_1 = beta_1
        
    
    def step(self, model_params):
        """
        Updates the model using the desired optimization algorithm (classic
        gradient descent, with optional momentum, or Adam)

        Parameters
        ----------
        model_params : list
            the parameters of the model, as returned by sequential.param().
            Each element is therefore a list [w, b, gradw, gradb]. Therefore,
            for each p in model_params, p[0] = w, p[1] = b, p[2] = gradw,
            p[3] = gradb

        """
        
        if self.Adam : #### Adam optimization ####
            
            if self.t == 0: # Initialization
                
                for p in model_params:
                    self.m.append([t.zeros((p[2].shape)), t.zeros((p[3].shape))])
                    self.v.append([t.zeros((p[2].shape)), t.zeros((p[3].shape))])
                    
                         
        
            self.t += 1
            for p, m, v in zip(model_params, self.m, self.v):
                
                m[0] = self.beta_1 * m[0] + (1-self.beta_1)*p[2]
                m[1] = self.beta_1 * m[1] + (1-self.beta_1)*p[3]
                
                v[0] = self.beta_2 * v[0] + (1-self.beta_2)*p[2].pow(2)
                v[1] = self.beta_2 * v[1] + (1-self.beta_2)*p[3].pow(2)
                
                m_hat_w = m[0]/(1-math.pow(self.beta_1, self.t))
                m_hat_b = m[1]/(1-math.pow(self.beta_1, self.t))
                v_hat_w = v[0]/(1-math.pow(self.beta_2, self.t))
                v_hat_b = v[1]/(1-math.pow(self.beta_2, self.t))
                
                p[0] -= self.alpha * t.div( m_hat_w, v_hat_w.sqrt() + self.eps)
                p[1] -= self.alpha * t.div( m_hat_b, v_hat_b.sqrt() + self.eps)
            return
                
        elif self.RMS :
            if self.t == 0: # Initialization
                for p in model_params:
                    self.v.append([t.zeros((p[2].shape)), t.zeros((p[3].shape))])
                    self.m.append([t.zeros((p[2].shape)), t.zeros((p[3].shape))])
                    self.phi.append([t.zeros((p[2].shape)), t.zeros((p[3].shape))])
                    
                     
            self.t += 1
            for p, v, m, phi in zip(model_params, self.v, self.m, self.phi):
                v[0] = self.nu * v[0] + (1-self.nu)*p[2].pow(2)
                v[1] = self.nu * v[1] + (1-self.nu)*p[3].pow(2)
                m[0] = self.nu * m[0] + (1-self.nu)*p[2]
                m[1] = self.nu * m[1] + (1-self.nu)*p[3]
                
                phi[0] =  self.beta_1 * phi[0] - self.psi * p[2].div((v[0] - m[0].pow(2)+ self.psi).sqrt())
                phi[1] =  self.beta_1 * phi[1] - self.psi * p[3].div((v[1] - m[1].pow(2)+ self.psi).sqrt())
                
                
                p[0] += phi[0]
                p[1] += phi[1]
            
        else: ### Gradient descent optimization ###
      
            if self.t == 0: # Initialization
                for p in model_params:
                    self.m.append([t.zeros((p[2].shape)), t.zeros((p[3].shape))])
                
            self.t += 1
            
            for p, m in zip(model_params, self.m):
                m[0] = self.momentum * m[0] - self.eta * p[2] 
                m[1] = self.momentum * m[1] - self.eta * p[3] 
                p[0] += m[0]
                p[1] += m[1] # With momentum = 0, that's "vanilla" SGD
            return
        
##############################################################################



class Sequential():
    """
    A class used to represent the whole neural network (a sequence of linear
                                                        and non-linear layers)
    
    Attributes
    ----------
    modules : list
        a list of all the layers of the network, each layer being defined with
        the appropriate class
        
    Methods
    -------
    forward(Input)
        performs forward pass
    
    backward(gradoutput)
        performs backward pass
    
    zero_grad
        sets dL/dw and dL/db to zero for all layers
            
    param
        returns list of parameters for the whole network
        
    re_init(optional parameters)
        re-initialize the whole network with the desired method
        
    """

    def __init__(self, *arg):
        self.modules = []
        for module in arg:
            self.modules.append(module)

    def forward(self, Input):
        """
        Performs forward pass
        
        Parameters
        ----------
        Input : torch tensor (a vector or a matrix depending on the mini-batch
                              setting)
            x_0, it's the input of the neural network
        
        Returns
        -------
        x_L : torch tensor (a vector or a matrix depending on the mini-batch
                            setting)
            output of the last layer
        
        """
        x = Input.clone() # x_0
        
        for module in self.modules:
            x = module.forward(x)
            
        return x # x_L
        
    def backward(self, dl_dx):
        """
        Performs backward pass
        
        Parameters
        ----------
        dl_dx : torch tensor (nb_samples_per_batch x output dimentions)
            the per sample derivative of the loss function with respect to x_L

        """
        
        temp = dl_dx

        for module in reversed(self.modules):
            temp = module.backward(temp)

        return

    def zero_grad(self):
        """
        Sets gradw and gradb to zero for all layers of the NN
        """
        for module in self.modules:
            if isinstance(module, Linear): # Only get linear layers
                module.zero_grad()

    

    def param(self):
        """
        Returns a list. Each element l of the list contains the parameters of
        layer l, [w, b, gradw, gradb]
        """
        params = []
        for module in self.modules:
            if isinstance(module, Linear): # Only get linear layers
                params.append(module.param())
        return params
          
    
    def re_init(self, Xavier=True, Gain = True, bias = True):
        """
        Reinitialises all the linear layers of the network with the desired
        method, either standard, Xavier or adjusted Xavier based on the
        activation functions.

        Parameters
        ----------
        Xavier : bool, optional
            Determines whether to use Xavier initialization. The default is True.
            (Since by default linear layers are initialized with the standard
            procedure, one is most likely to call this method if Xavier is
            desired)
            
        Gain : bool, optional
            Detetermines whether to adjust Xavier's initialization based on the
            activation functions. The default is True.
        
        bias : bool, optional
            Determines whether to have a bias term. The default is True.
 
        """
        nb_layers = len(self.modules)
        
        for module, i in zip(self.modules, range(nb_layers)):
            
            if isinstance(module, Linear): 
 
                if Gain and i != nb_layers - 1: # (check if it's the last
                                                #  layer)
                    g = self.modules[i+1].gain()
                else:
                    g = 1
                     
                module.__init__(module.in_dim, module.out_dim, bias = bias,
                                Xavier = Xavier, gain = g)

        return
    

    
    
    
    
    
    
    
################# Here the loss functions are defined:  ######################



class LossMSE():
    """
    A class used to represent the Mean Square Error loss function
    ...
    
    Attributes
    ----------
    pred : torch tensor (nb_samples x D)
        predicted output
        
    targ : torch tensor (nb_samples x D)
        contains the nB_samples target vectors
    
    nb_samples : scalar
        either one (batch gradient descent) or the size of the minibatch
        (minibatch gradient descent)
        
    dim_target : scalar
        dimentionality of the target vectors (=D). >1 if one hot encoding was
        used
        
        
    Methods
    -------
    compute(pred, target)
        computes the per mini-batch MSE loss
    
    grad()
         returns the per sample gradient of the loss function with respect to x_L
    
    
        
    """
    def __init__(self):
        self.targ = None
        self.pred = None
        self.nb_samples = None
        self.dim_target = None
    
    

    def compute(self, pred, target):
        """
        Computes the per sample (or per mini-batch) MSE loss, and updates all
        the class attributes
        
        Parameters
        ----------
        pred : torch tensor (nb_samples_per_batch x D)
            predicted output
        
        targ : torch tensor (nb_samples_per_batch x D)
            target vector
            
        Returns
        -------
        loss : float
            
        
        """
        self.pred = pred.clone()
        self.nb_samples, self.dim_target = self.pred.size()
                
        if target.dim() == 1: # We want to deal with 2D tensors, as self.pred
             self.targ = target.clone()[:,None]
        else:
            self.targ = target.clone()

        loss = (pred - target).pow(2).mean()
        
        return loss


    def grad(self):
        """
        Returns the per sample or per mini-batch gradient of the loss function
        with respect to the predicted output
        """
                       
        dl_dxL = 2*(self.pred- self.targ)/(self.nb_samples*self.dim_target)
        
        return dl_dxL


#############################################################################


class BCELoss():
    """
    BCE Losses 
    
    To use this function the target function should have a value between zero and one.
    
    """

    def __init__(self):
        self.pred = None
        self.targ = None
        self.nb_samples = None
        self.min = 1e-12

    def compute(self,pred, target):

        self.pred = pred.clone()
        
        self.nb_samples = self.pred.shape[0]
    

        if target.dim() == 1:
             self.targ = target.clone()[:,None]
        else:
            self.targ = target.clone()
            
       
        loss = -self.targ*t.log(self.pred.clamp(min = self.min, max = 1)) - (1-self.targ)*t.log((1-self.pred).clamp(min = self.min, max = 1))
        
        return loss.mean()
    

    def grad(self):
        
        
        dl_dxL = -self.targ/(self.pred.clamp(min = self.min, max = 1)) + (1-self.targ)/((1-self.pred).clamp(min = self.min, max = 1))
        
        return dl_dxL/self.nb_samples ## maybe useful because we took the mean across the samples

################################################################################


