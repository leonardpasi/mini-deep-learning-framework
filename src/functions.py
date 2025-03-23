### Here we import the different modules : 

import Model_NN as nn
from models import *
import math
import time
import torch as t
t.set_grad_enabled(False)



# We will import directly the professor data generating function and other
# useful functions #######################################################


def generate_disc_set(nb):
    """
    Generates a set of nb points sampled uniformly in [0, 1] , each with a
    label 0 if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2π), 
    and 1 inside.
    
    Parameters
    ----------
    nb : int
        number of datapoints to generate
        
    Returns
    -------
    dataset : torch tensor (nb x 2)
    
    target : torch vector (nb)
        datatype is int
        
    """
    dataset = t.empty(nb, 2).uniform_(-0.5, 0.5)
    target = dataset.pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).int()
    dataset.add_(0.5)
 
    return dataset, target


def normalize_data(data_set):
    """
    Normalizes a dataset

    Parameters
    ----------
    data_set : torch tensor (nb_samples x nb_dim)
        
    Returns
    -------
    normalized : torch tensor (nb_samples x nb_dim)

    """
    mean, std = data_set.mean(0), data_set.std(0)
    normalized = (data_set-mean)/std
    
    return normalized



def one_hot_encoding(Y):
    """
    Performs one hot encoding

    Parameters
    ----------
    Y : torch vector (nb_samples)
        contains the labels, coded as consecutive integers (zero to nb of
                                                            classes - 1)

    Returns
    -------
    Y_hot : torch tensor (nb_sample x nb_classes)
       
    """
    
    nb_samples = Y.shape[0]
    nb_classes = Y.max()+1
    if nb_classes == 1:
        print("Should have at least two classes!")
        
    Y_hot = t.zeros((nb_samples, nb_classes))
    
    for i in range(nb_classes):
        Y_hot[Y == i, i] = 1
    
    return Y_hot
    
   

    
def testing_across_parameters(X_train, Y_train, X_test, Y_test, nb_folds, epochs, mini_batch_size, active_func, criterion, list_parameter,
                               adam = False, rms = False, momentum = 0 , xavier = False, gain = False, BCE = False, message_folds = True, losses = False, shuffle = True ): 
    """
    
    Execute the training using model_train function defined below.
    Permit to compare different attributes of our framework.

    """
    optimizer = nn.Optimizer(eta = 1e-1, Adam = adam, RMS = rms,  momentum= momentum)
    nb_samples = X_train.shape[0]
    if BCE:
        model = model_compare_BCE_MSE()
    else:
        model = model_active_function(active_func)
    train_err_runs = []
    train_err_ep = [0]*epochs
    test_err_ep = [0]*epochs
    train_loss_ep = [0]*epochs
    test_loss_ep = [0]*epochs
    test_err_runs = []
    avg_time = 0
    for k in range(nb_folds):
        model.re_init(Xavier = xavier, Gain = gain) # Re-initialize
        
        time, train_loss_epochs, train_err_epochs, test_loss_epochs,test_err_epochs = model_train(model, optimizer, criterion,X_train , 
                                                            Y_train,X_test, Y_test,
                                                            mini_batch_size, nb_epochs = epochs,
                                                            verbose = False, testing = True, shuffle = shuffle)  # Training the model and computing the error
        
        train_err_ep = [(train_err_ep[i] + train_err_epochs[i]/nb_folds) for i in range(epochs)] ## Compute the mean for all the runned epochs
        test_err_ep = [(test_err_ep[i] + test_err_epochs[i]/nb_folds) for i in range(epochs)]
        if losses: 
            train_loss_ep = [(train_loss_ep[i] + train_loss_epochs[i]/nb_folds) for i in range(epochs)]
            test_loss_ep = [(test_loss_ep[i] + test_loss_epochs[i]/nb_folds) for i in range(epochs)]
        
        train_err_runs.append(train_err_epochs[-1])
        test_err_runs.append(test_err_epochs[-1])
        
            
        avg_time += time/nb_folds
    
    stats(train_err_runs, test_err_runs, nb_folds, nb_samples, avg_time) 
    if list_parameter != None and message_folds:
        plot_results(train_err_ep,test_err_ep, message = list_parameter)
    
    if losses:
        return train_err_ep, test_err_ep, train_loss_ep, test_loss_ep
    else:
        return train_err_ep, test_err_ep
    
    
    
    

##############################################################################
def model_train(model, optimizer, criterion, train_input, train_target,test_input, test_target,
                       mini_batch_size = 100, nb_epochs = 100 , verbose = False, testing = False, shuffle = True):
    """
        Trains the model, using the specified optimizer and the specified update
        rate strategy
        
        Parameters
        ----------
        model : Sequential
            the model to be trained
            
        optimizer : Optimizer
            the optimizer to be used
        
        criterion : LossMSE or BCEloss
            the loss function to be used
        
        train_input : torch tensor (nb of samples x d)
            feature vectors for training
            
        train_target : torch tensor (nb of samples x d)
        
        test_input : torch tensor (nb of samples x d)
            feature vectors for testing
            
        test_targt : torch tensor (nb of samples x d)
          
            
        mini_batch_size : int, optional
            Set to 100 by default
    
        nb_epochs : int, optional
            self explanatory. Set to 100 by default
        
        Verbose : bool, optional
            determines whether to print the accumulative error and loss.
        
        Testing: bool, optional
            determines whether we execute also the testing.
        
            
        """
    train_loss_epochs = []
    train_err_epochs = []
    test_loss_epochs = [] 
    test_err_epochs = [] 
    time_f = 0
    for epoch in range(nb_epochs):
        if (not shuffle):
            train_data = zip(train_input.split(mini_batch_size),train_target.split(mini_batch_size))
            test_data = zip(test_input.split(mini_batch_size),test_target.split(mini_batch_size))
        else:
            premute_train = torch.randperm(len(train_input))
            permute_test = torch.randperm(len(test_input))
            train_data =  zip(train_input[premute_train].split(mini_batch_size),
                             train_target[premute_train].split(mini_batch_size))
            test_data = zip(test_input[permute_test].split(mini_batch_size),
                            test_target[permute_test].split(mini_batch_size))
             
        start = time.perf_counter() 
        train_loss, train_error = training_loop(model, train_data, optimizer, criterion)
        end = time.perf_counter()
        time_f += end-start # computing the training  time across the epochs
        train_loss_epochs.append(train_loss.item()) 
        train_err_epochs.append(train_error.item())
        if verbose :
            print("Epoch = {}\n\t train_error = {}".format(epoch,train_error))
        if testing :
            test_loss, test_error = validation_loop(model,test_data,criterion)
            test_loss_epochs.append(test_loss.item()) 
            test_err_epochs.append(test_error.item()) 
            if verbose:
                print("\t test_error = {1}".format(epoch,test_error))
                
    if testing:
        return time_f, train_loss_epochs, train_err_epochs, test_loss_epochs, test_err_epochs
    else:
        return time_f, train_loss_epochs, train_err_epochs






################Function to train the model with #####################

def training_loop(model,train_data,optimizer,criterion):
    
    """
    Training loop used to train the model and compute the training losses and errors 
    
    """
     
    counter = 0 
    nb_errors = 0 
    acc_loss = 0
  
    for inputs, targets in train_data: 
        counter += len(inputs)
        model.zero_grad()
        
        pred = model.forward(inputs)
        loss = criterion.compute(pred,targets)
        
        model.backward(criterion.grad())
        
        optimizer.step(model.param())
        
        
        acc_loss += loss
        nb_errors += get_error(pred,targets)
            
    error_rate = nb_errors / counter
    avg_loss = acc_loss / counter
    return avg_loss,error_rate    



#### Function to validate with on the test data ######################
def validation_loop(model,test_data, criterion):  
    
    """

    Execute a forward propagation to compute the prediction made at the end.
    Returns: The computed error and losses.

    """
    
    nb_errors = 0
    acc_loss = 0
    counter = 0
    for inputs, targets in test_data:
        counter += len(inputs)
        pred = model.forward(inputs)
        nb_errors += get_error(pred,targets)
        loss = criterion.compute(pred,targets)
        acc_loss += loss
    error_rate = nb_errors / counter
    avg_loss = acc_loss / counter
    return avg_loss,error_rate



def get_error(pred ,targ):
    
    """
    
    Function to get the errors when comparing prediction and target labels.
    
    """
    
    if pred.shape[1] == 1:
        thr = 0.5
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        nb_errors = sum([label != target for (target,label) in zip(targ,pred)])
        
        
    else:
       p,labels = t.max(pred,1)
       nb_errors = sum([label != target for (target,label) in zip(targ,labels)])

    return nb_errors





def compute_nb_error(model, X, Y, mini_batch_size):
    """
    Computes number of errors. Winner takes all approach is used. If it's a
    multi-class problem, one-hot encoding must be used. If it's a binary
    classification problem, one-hot encoding is not necessary.
    
    Parameters
    ----------
    model : Sequential
        the model that has been trained
    
    X : torch tensor (nb of samples x d)
        feature vectors
        
    Y_hot : torch tensor (nb of samples x nb_classes)
        labels, hot-encoded
        
    mini_batch_size : int
        self explanatory

    
    Returns
    -------
    errors : int
        number of errors between the predicted and the actual label        
    """
    
    errors = 0
        
    for b in range(0, X.size(0), mini_batch_size):
        
        pred = model.forward( X.narrow(0, b, mini_batch_size))
        
        pred = process_pred(pred)
        
        for k in range(mini_batch_size):
            if not (t.all( Y[b+k].eq( pred[k] ))):
                errors += 1
                
    return errors

def process_pred(pred):
    
    thr = 0.5
    
    if pred.shape[1] == 1: # If target vectors are actually scalars, i.e.
                              # one hot encoding has not been used, we threshold
                              
        pred[pred < thr] = 0
        pred[pred > thr] = 1
            
    else: # Winner takes all
        
        idx = pred.argmax(1)
        pred_w = t.zeros((pred.shape))
        pred_w[range(pred.shape[0]),idx] = 1
        pred = pred_w
                
    return pred


def stats(train_errors, test_errors, folds, nb_samples, avg_time, message = "our framework", std = False):
    
    print("\nStatistical Analysis over {:d} folds for {} :".format(folds,message))
    print("------------------------------------")
    
    mean_train = sum(train_errors)/len(train_errors)
    mean_test = sum(test_errors)/len(test_errors)
    if std:
        sd_train = sum((x-mean_train)**2 for x in train_errors) / len(train_errors)
        sd_test = sum((y-mean_test)**2 for y in test_errors) / len(test_errors)
    
    if std : 
        print("                              Average:             Standard deviation:")
        print("Error  on training set:     {:0.3f}               {:0.7f} ".format(mean_train, sd_train))
        print("Error on testing set :     {:0.3f}                {:0.7f}".format(mean_test, sd_test))
    else: 
        print("                              Average:         ")
        print("Error  on training set:     {:0.3f}            ".format(mean_train))
        print("Error on testing set :     {:0.3f}             ".format(mean_test))
    print("The  average time to train and perdict: {:0.3f}".format(avg_time))
    print("------------------------------------")
    if std:
        return mean_train, sd_train, mean_test, sd_test
    else: 
        mean_train, mean_test
    

    