### This file is inspired from our first project and will implement similar functions

import torch 

import math 
import matplotlib.pyplot as plt
from functions import *
import time

from torch import nn 
from torch.nn import functional as F


torch.set_grad_enabled(True)






def basic_model(input_dim = 2, output_dim  = 1, nb_hidden = 25, active_func = nn.ReLU()):
    return nn.Sequential(nn.Linear(input_dim,nb_hidden),
                           active_func,
                           nn.Linear(nb_hidden,nb_hidden),
                           active_func,
                           nn.Linear(nb_hidden,nb_hidden),
                           active_func,
                           nn.Linear(nb_hidden,nb_hidden),
                           active_func,
                           nn.Linear(nb_hidden,output_dim))


########################################################
## We declare the functions used to train on pytorch #####
####################################################


def testing_on_pytorch(nb_folds, epochs , X_train,Y_train, X_test, Y_test, SGD = True ,crit = True, losses = False, message_folds = True,
                       shuffle= True):
    """
    This function is useful to train and test the prediction using pytorch :
    
    """
    train_err_runs = []
    test_err_runs = []
    train_err_ep =  [0]*epochs
    test_err_ep = [0]*epochs
    train_loss_ep =[0]*epochs
    test_loss_ep = [0]*epochs
    avg_time = 0
    nb_samples = X_train.shape[0]
    
    if crit:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    
    for i in range(nb_folds):
        
        model = basic_model()
        if SGD:
            optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        
        time, train_loss_epochs, train_err_epochs,test_loss_epochs, test_err_epochs = py_train_model(model,
                    X_train, Y_train, X_test, Y_test,criterion, optimizer, nb_epochs = epochs,  verbose = False,
                    testing = True, shuffle = shuffle)
      
        train_err_runs.append(train_err_epochs[-1])
        test_err_runs.append(test_err_epochs[-1])
        
        train_err_ep = [(train_err_ep[i] + train_err_epochs[i]/nb_folds) for i in range(epochs)]
        test_err_ep = [(test_err_ep[i] + test_err_epochs[i]/nb_folds) for i in range(epochs)]
        if losses: 
            train_loss_ep = [(train_loss_ep[i] + train_loss_epochs[i]/nb_folds) for i in range(epochs)]
            test_loss_ep = [(test_loss_ep[i] + test_loss_epochs[i]/nb_folds) for i in range(epochs)]
        
     
        avg_time += (time)/nb_folds
    if message_folds :
        stats(train_err_runs, test_err_runs, nb_folds, nb_samples, avg_time,  message = "Pytorch framework") 
    if losses:
        return train_err_ep, test_err_ep, train_loss_ep, test_loss_ep
    else:     
        return train_err_ep, test_err_ep











##### this function will execute the trying implementing pytorch :

def py_train_model(model, train_input, train_target,test_input, test_target,
                 criterion,optimizer,  mini_batch_size = 100, nb_epochs = 100 , verbose = False, testing = False, shuffle = True):

    train_loss_epochs = []
    train_err_epochs = []
    test_loss_epochs = [] 
    test_err_epochs = [] 
    time_f = 0
    for epoch in range(nb_epochs):
        if ( not shuffle):
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
        train_loss, train_error = training_loop(model,train_data,optimizer,criterion = criterion)
        end = time.perf_counter()
        time_f += (end-start) # Computing the time to execute all the training throught the epochs
        
        train_loss_epochs.append(train_loss.item()) 
        train_err_epochs.append(train_error.item())
        if verbose :
            print("Epoch = {}\n\t train_error = {}".format(epoch,train_error))
        if testing :
            test_loss, test_error = validation_loop(model,test_data, criterion= criterion)
            test_loss_epochs.append(test_loss.item()) 
            test_err_epochs.append(test_error.item()) 
            if verbose:
                print("\t test_error = {}".format(test_error))
                
    if testing:
        return time_f, train_loss_epochs, train_err_epochs, test_loss_epochs,  test_err_epochs 
    else:
        return time_f, train_loss_epochs, train_err_epochs     
   
    
#######################################################

def get_error(pred ,targ):
    
    if pred.shape[1] == 1:
        thr = 0.5
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        nb_errors = sum([label != target for (target,label) in zip(targ,pred)])
        
        
    else:
        p,labels = torch.max(pred,1)
        nb_errors = sum([label != target for (target,label) in zip(targ,labels)])

    return nb_errors

###########################################################

########################################################

def training_loop(model, train_data, optimizer, criterion):   # inspired form our first project functions
                                                                            # modified to be implemented on our data                     
    model.train()   # Indicate to the model that you are training it
    
    counter = 0 
    nb_errors = 0 
    acc_loss = 0
    
    for inputs, targets in train_data: 
        counter += len(inputs)
        size_batch = len(inputs)
        optimizer.zero_grad()
        output = model(inputs)
        
        loss = criterion(output.squeeze(),targets.float())
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            acc_loss += loss
            nb_errors += get_error(output,targets)
            
    error_rate = nb_errors / counter
    avg_loss = acc_loss / counter
    return avg_loss,error_rate    
    

########################################################################    
def validation_loop(model,test_data, criterion):  
    
    model.eval()
    with torch.no_grad():  #in the evaluation mode we should not update the weights
        nb_errors = 0
        acc_loss = 0
        counter = 0
        for inputs, targets in test_data:
            counter += len(inputs)
            output = model(inputs)
            loss = criterion(output.float(),targets.float())
            nb_errors += get_error(output,targets)
            acc_loss += loss
    error_rate = nb_errors / counter
    avg_loss = acc_loss / counter
    return avg_loss,error_rate
####################################################################### 






#####Plot function taking into consideration losses and error #############

def  plot_results_loss_err(train_loss_epochs,test_loss_epochs, train_err_epochs, test_err_epochs):
    
    fig,ax = plt.subplots(1,2)
    ax[0].plot(train_loss_epochs)
    ax[0].plot(test_loss_epochs)
    ax[0].set_ylabel("Loss :")
    ax[0].set_xlabel("Epochs :")
    ax[1].plot(train_err_epochs)
    ax[1].plot(test_err_epochs)
    ax[1].set_ylabel("Error rate:")
    ax[1].set_xlabel("Epochs :")
    plt.show()



