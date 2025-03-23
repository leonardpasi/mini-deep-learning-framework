## This module contains the different architectures we will test with our framework:
    
import Model_NN as nn
import Activation_functions as ac
import torch
torch.set_grad_enabled(False)




# Test different activation functions :

def model_active_function(active_func, input_dim = 2 ,output_dim  = 1 , nb_hidden = 25):
    
    if type(active_func) == type(ac.Tanh()):
        return nn.Sequential(nn.Linear(input_dim,nb_hidden), active_func,
                      nn.Linear(nb_hidden,nb_hidden), active_func,
                      nn.Linear(nb_hidden,nb_hidden), active_func,
                      nn.Linear(nb_hidden,nb_hidden), active_func,
                      nn.Linear(nb_hidden,output_dim), ac.Relu())
    
    else:
        return nn.Sequential(nn.Linear(input_dim,nb_hidden), active_func,
                      nn.Linear(nb_hidden,nb_hidden), active_func,
                      nn.Linear(nb_hidden,nb_hidden), active_func,
                      nn.Linear(nb_hidden,nb_hidden), active_func,
                      nn.Linear(nb_hidden,output_dim), active_func)


def model_different_active_func(active_func_1 = ac.Relu(), active_func_2= ac.ELU(), input_dim = 2 ,
                                output_dim  = 1 , nb_hidden = 25):
    
    return nn.Sequential(nn.Linear(input_dim,nb_hidden), active_func_1,
                      nn.Linear(nb_hidden,nb_hidden), active_func_2,
                      nn.Linear(nb_hidden,nb_hidden), active_func_2,
                      nn.Linear(nb_hidden,nb_hidden), active_func_2,   
                      nn.Linear(nb_hidden,output_dim), active_func_1)



def model_compare_BCE_MSE(active_func_1 = ac.Relu(), active_func_2= ac.ELU() , active_func_3 = ac.Sigmoid(), input_dim = 2 ,
                                output_dim  = 1 , nb_hidden = 25):  
    return nn.Sequential(nn.Linear(input_dim,nb_hidden), active_func_1,
                      nn.Linear(nb_hidden,nb_hidden), active_func_1,
                      nn.Linear(nb_hidden,nb_hidden), active_func_2,
                      nn.Linear(nb_hidden,nb_hidden), active_func_2,   
                      nn.Linear(nb_hidden,output_dim),active_func_3)
    
    



    
    