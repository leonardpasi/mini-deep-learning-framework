"""
This module contains the definitions of the classes for the activation
functions. Four activation functions were implemented: Relu, tanh, LeakyRelu
and ELU.

"""

import math

##############################################################################

class Relu():
    """
    A class used to represent an activation layer using ReLu, i.e. a layer
    where the transformation s |--> x = Relu(s) takes place. The function is
    applied component wise, such that x and s have the same size, typically
    (nb_samples_per_batch x out_dim)
    ...
    
    Attributes
    ----------
    s : torch tensor
        summation variable
        
    Methods
    ------
    forward(Input)
        performs forward pass
    
    backward(gradoutput)
        performs backward pass
    
    gain
        returns the recommended gain value for Relu
    
    """
    
    def __init__(self):
        self.s = None


    def forward(self, Input):
        """
        Performs forward pass and updates s
        
        Parameters
        ----------
        Input : torch tensor
            the new value for class attribute s
        
        Returns
        -------
        result : torch tensor
            result of the forward pass x = Relu(s)
        
        """
        
        self.s = Input.clone() # Update s
        result = Input.clone()
        result[result < 0 ] = 0

        return result

    def backward(self, dl_dx):
        """
        Performs backward pass. Should not be called unless forward method has
        been called once before
        
        Parameters
        ----------
        dl_dx : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to x. Should have same dimentions as attribute s.
        
        Returns
        -------
        dl_ds : torch tensor ((nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to s
            = Relu'(s) * dl_dx, where '*' is the element-wise multiplication
        
        """
        
        temp = self.s.clone()
        temp[temp < 0 ] = 0
        temp[temp >= 0 ] = 1

        return temp.mul(dl_dx)
    
    def gain(self):
        """
        Returns the recommended gain value for Relu
        """
        return math.sqrt(2)

##############################################################################

class Tanh():
    """
    A class used to represent an activation layer using hyperbolic tangent, 
    i.e. a layer where the transformation s |--> x = tanh(s) takes place.
    The function is applied component wise, such that x and s have the same
    size, typically (nb_samples_per_batch x out_dim)
    ...
    
    Attributes
    ----------
    s : torch tensor
        summation variable
        
    Methods
    ------
    forward(Input)
        performs forward pass
    
    backward(gradoutput)
        performs backward pass
    
    gain
        returns the recommended gain value for tanh
    
    """
    
    def __init__(self):
        self.s = None
    
    def forward(self, Input):
        """
        Performs forward pass and updates s
        
        Parameters
        ----------
        Input : torch tensor
            the new value for class attribute s
        
        Returns
        -------
        result : torch tensor
            result of the forward pass x = tanh(s)
        
        """
          
        self.s = Input.clone() # Update s

        return self.s.tanh()

    def backward(self, dl_dx):
        """
        Performs backward pass. Should not be called unless forward method has
        been called once before
        
        Parameters
        ----------
        dl_dx : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to x. Should have same dimentions as attribute s.
        
        Returns
        -------
        dl_ds : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to s
            = tanh'(s) * dl_dx, where tanh'(s) = 1 - tanh^2(s) and '*' is
            the element-wise multiplication
        
        """

        s = self.s

        return (1-s.tanh().pow(2)).mul(dl_dx)
    
    def gain(self):
        """
        Returns the recommended gain value for tanh
        """
        return 5/3



##############################################################################




class LeakyRelu():
    """
    A class used to represent an activation layer using LeakyRelu, 
    i.e. a layer where the transformation s |--> x = LeakyRelu(s) takes place.
    The function is applied component wise, such that x and s have the same 
    size, typically (nb_samples_per_batch x out_dim)
    ...
    
    Attributes
    ----------
    s : torch tensor
        summation variable
    a : scalar
        gradient of the LeakyRelu function for x<0. Set by default to 0.01
        
    Methods
    ------
    forward(Input)
        performs forward pass
    
    backward(gradoutput)
        performs backward pass
    
    slope_def(slope)
        sets attribute 'a' to slope
        
    gain
        returns the recommended gain value for LeakyRelu
    
    """
    
    def __init__(self, slope = 0.01):
        """
        Initializes the LeakyRelu with the desired negative slope. The default
        is 0.01.
        """
        
        self.s = None
        self.a = slope

    def forward(self, Input):
        """
        Performs forward pass and updates s
        
        Parameters
        ----------
        Input : torch tensor
            the new value for class attribute s
        
        Returns
        -------
        val : torch tensor
            result of the forward pass x = LeakyRelu(s)
        
        """
        self.s = Input.clone() # Update s
        
        val = Input.clone()
        val[val < 0] *= self.a

        return  val

    def backward(self, dl_dx):
        """
        Performs backward pass. Should not be called unless forward method has
        been called once before.
        
        Parameters
        ----------
        dl_dx : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to x. Should have same dimentions as attribute s.
        
        Returns
        -------
        dl_ds : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to s
            = LeakyRelu'(s) * dl_dx, where '*' is the element-wise
            multiplication
        
        """

        temp = self.s.clone()
        temp[temp < 0] = self.a
        temp[temp >= 0] = 1

        return temp.mul(dl_dx)

    def slope_def(self, slope):
        """
        Defines the negative slope with the desired value
        """
        self.a = slope
    
    def gain(self):
        """
        Returns the recommended gain value for LeakyRelu
        """
        return math.sqrt(2/(1+pow(self.a,2)))



##############################################################################




class ELU():
    """
    A class used to represent an activation layer using Exponential Linear
    Unit (ELU),
    i.e. a layer where the transformation s |--> x = ELU(s) takes place.
    The function is applied component wise, such that x and s have the same 
    size, typically (nb_samples_per_batch x out_dim)
    ...
    
    Attributes
    ----------
    s : torch tensor
        summation variable
    a : scalar
        parameter of the ELU function. The default is 0.1
        
    Methods
    ------
    forward(Input)
        performs forward pass
    
    backward(gradoutput)
        performs backward pass
    
    const_def(c)
        sets attribute 'a' to c
        
    show_const
        returns a
        
    gain
        returns the recommended gain value for ELU (Not known! Returns 1)
    
    """

    def __init__(self, const = 0.1):
        """
        Initializes ELU with the desired parameter. The default is 0.01.
        """
        self.s = None
        self.a = const

    def forward(self, Input):
        """
        Performs forward pass and updates s
        
        Parameters
        ----------
        Input : torch tensor
            the new value for class attribute s
        
        Returns
        -------
        val : torch tensor
            result of the forward pass x = ELU(s)
        
        """
        self.s = Input.clone() # Update s
        
        val = Input.clone()
        val[val < 0 ] = self.a*(val[val < 0].exp() - 1 )

        return val

    def backward(self, dl_dx):
        """
        Performs backward pass. Should not be called unless forward method has
        been called once before.
        
        Parameters
        ----------
        dl_dx : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to x. Should have same dimentions as attribute s.
        
        Returns
        -------
        dl_ds : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to s
            = ELU'(s) * dl_dx, where '*' is the element-wise multiplication
        
        """

        temp = self.s.clone()
        temp[temp >= 0 ] = 1
        temp[temp < 0] = self.a * (temp[temp < 0].exp())

        return temp.mul(dl_dx)

    
    def const_def(self, c):
        """
        Sets the parameter of the ELU function to the desired value
        """
        self.a = c

    def show_const(self):
        """
        Returns the current parameter of the ELU function
        """
        return self.a
    
    def gain(self):
        """
        Returns the recommended gain value for ELU (Not known! Returns 1)
        """
        return 1. # Not known for ELU

    



class Sigmoid():
    
    def __init__(self):
                

        self.s = None
        self.val = None
        
    def forward(self, Input):
        """
        Performs forward pass and updates s
        
        Parameters
        ----------
        Input : torch tensor
            the new value for class attribute s
        
        Returns
        -------
        val : torch tensor
            result of the forward pass x = Sigmoid(s)
        
        """
        
        self.s = Input.clone()
        
        self.val = 1 / (1 + (-self.s).exp())
        return self.val
    
    def backward(self, dl_dx):
        
        """
        Performs backward pass. Should not be called unless forward method has
        been called once before.
        
        Parameters
        ----------
        dl_dx : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to x. Should have same dimentions as attribute s.
        
        Returns
        -------
        dl_ds : torch tensor (nb_samples_per_batch x out_dim)
            the gradient of the per-sample or per-mini-batch loss with respect 
            to s
            = Sigmoid(s) * dl_dx, where '*' is the element-wise multiplication
        
        """
        
        temp = self.val.mul(1-self.val)
        
        
        return  temp.mul(dl_dx)

    
    def gain(self):
        """
        Returns the recommended gain value for Sigmoid
        """
        return 1. 
        
    
        
    


