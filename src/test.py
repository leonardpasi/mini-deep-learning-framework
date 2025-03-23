import torch

from functions import * 
from models import *
from plots import *
import Model_NN as nn
import Activation_functions as ac
import functions_pytorch as py
import time
## We will import this library on order to create some plots : 
torch.set_grad_enabled(False)


# Generate the data
nb_samples = 1000
X_train, Y_train = generate_disc_set(nb_samples)
X_test, Y_test = generate_disc_set(nb_samples)

# Generate the data
nb_samples = 1000
X_train, Y_train = generate_disc_set(nb_samples)
X_test, Y_test = generate_disc_set(nb_samples)

### Display the data
#plot_graph(X_train, Y_train, "training")
#plot_graph(X_test, Y_test, "testing")

# Normalize the datasets:
X_train_n = normalize_data(X_train)
X_test_n = normalize_data(X_test)

# Define the batch size
mini_batch_size = 100
epochs = 80

# Define number of folds and error tensors
nb_folds = 10

## The architecture dimensions: 
    
input_dim = 2
output_dim  = 1
nb_hidden = 25

## Here we define if we want to plot the different results: 
plots = True

###############################################################################
# Compare here pytorch with our framework : having the same intial conditions
###############################################################################

# We will try to test the accuracies of the two models together:

## here we execute the test on our framework : 
results = []
criterion = nn.LossMSE()
print("We compare here between our framework and pytorch: ")
train_err_ep, test_err_ep = testing_across_parameters(X_train_n, Y_train, X_test_n, Y_test, nb_folds, epochs, mini_batch_size, 
                                                          ac.Relu(), criterion,None, adam = True , shuffle = False)
results.append(train_err_ep)
results.append(test_err_ep)
torch.set_grad_enabled(True)  # Here we need  to set back the gradient to True in order to use pytorch 
train_err_ep_py, test_err_ep_py = py.testing_on_pytorch(nb_folds,epochs, X_train_n,Y_train, X_test_n, Y_test, crit = True, SGD = False, shuffle = True)
torch.set_grad_enabled(False)
results.append(train_err_ep_py)
results.append(test_err_ep_py)

#plot_results(train_err_ep,test_err_ep, message = 'on our Own Framework')
#plot_results(train_err_ep_py,test_err_ep_py, message = 'on pytorch Framework')
if plots: 
    plot_framework_compare(results,epochs,)
    plot_framework_compare(results,epochs, zoom = 50)
    
    
    
####################################################################################
### Here we procced to the testing of all activation functions and compare them : 
####################################################################################



# Those are the existing activation functions in our framework :
list_func = ['Relu', 'Elu' , 'LeakyRelu', 'Tanh']    
criterion = nn.LossMSE()
results_train = []
results_test = []
for list_func, active_func in zip(list_func,[ac.Relu(), ac.ELU(), ac.LeakyRelu(), ac.Tanh()]): 
    print("Those are the  mean results  for an architecture with the activation function {} across {} folds: ".format(list_func, nb_folds))
    train_err_ep, test_err_ep = testing_across_parameters(X_train_n, Y_train, X_test_n, Y_test, nb_folds, epochs, mini_batch_size, 
                                                          active_func, criterion, list_func,message_folds = False, shuffle = True)
    results_train.append(train_err_ep)
    results_test.append(test_err_ep)
if plots:
    act_opt_functions_plot(results_train,results_test,epochs )
    act_opt_functions_plot(results_train,results_test,epochs, zoom = 50)


#########################################################################################
#### Here we are going to compare different optimizer and their impact on our Neural Net:
##########################################################################################

results_train = []
results_test = []
criterion = nn.LossMSE()
# Those are the different optimizers implemented in our framework :
list_optim = ['Vanilla', 'SGD with momentum', 'RMS' , 'Adam']

for list_optim, adam , rms,  momentum in zip(list_optim, [False, False, False, True], [False, False,True,False] ,  [0,0.2,0,0]):
    print("Those are the mean results  for a training with the optimizer {} across {} folds: ".format(list_optim, nb_folds))
    train_err_ep, test_err_ep = testing_across_parameters(X_train_n, Y_train, X_test_n, Y_test, nb_folds, epochs, mini_batch_size, 
                                                          ac.Relu(), criterion, list_optim,adam = adam, gain = False, 
                                                          rms = rms, momentum = momentum, message_folds = False, shuffle = False)
    results_train.append(train_err_ep)
    results_test.append(test_err_ep)
if plots:
    act_opt_functions_plot(results_train,results_test, epochs, act_opt = True)
    act_opt_functions_plot(results_train,results_test,epochs, act_opt = True, zoom = 50)

    
    
    
########################################################################      
### Comparing the different Loss functions implemented on the framework: 
########################################################################

nb_folds = 10
epochs = 100
results_train = []
results_test = []
#Those are the different loss functions : 
list_criterion = ["MSELoss", "BCELoss"]
for list_criterion, criterion in zip(list_criterion,[nn.LossMSE(), nn.BCELoss()]):
    print("Those are the mean  for a training using the criterion {} across {} folds: ".format(list_criterion, nb_folds))
    train_err_ep, test_err_ep = testing_across_parameters(X_train_n, Y_train, X_test_n, Y_test, nb_folds, epochs, mini_batch_size, 
                                                          ac.Relu(), criterion, list_criterion, rms = True, message_folds = False, BCE = True)
    results_train.append(train_err_ep) 
    results_test.append(test_err_ep)
if plots:
    act_opt_functions_plot(results_train,results_test, epochs,  loss = True)
    act_opt_functions_plot(results_train,results_test,epochs, loss = True, zoom = 50)    
