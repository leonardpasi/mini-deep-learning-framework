
import matplotlib.pyplot as plt


#############################################################################################
###This file is useful to have all the plots necessary to discuss the framework performances
############################################################################################


def plot_graph(data, target, label):
    
    """
    Display the data set distributions and the labels assigned to each sample
    
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    plt.scatter(data[target == 1,0],data[target == 1,1])
    plt.scatter(data[target == 0,0], data[target == 0,1])
    circle = plt.Circle( (0.5,0.5), 1/(math.sqrt(2* math.pi)), fill = False)
    ax.add_patch(circle)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("Scatter of the {} set :".format(label))
    plt.show()
    return



def plot_results(train_err_epochs,test_err_epochs, message = 'no_message'):
    """
    Display the accuracy of the model across the given number of epochs 
    
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 20
    plt.figure()
    plt.plot(train_err_epochs, label='train')
    plt.plot(test_err_epochs, label = 'test')
    plt.legend()
    
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    """
    if message == 'no_message':
        plt.title("The predictions errors across epochs: ")
    else:
        plt.title("The predictions errors across epochs for tested {}: ".format(message))
        
    """
    plt.show()
    
#######################################


def plot_framework_compare(results,epochs, zoom = None, file_name = "no_name" ):
    
    """
    Plot and compare the two framework perfomances with similar initial conditions:
        
    """
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 20
    list_label = ["Framework train", "Framework test", "Pytorch train", "Pytorch test"]
    
    plt.figure()
    for i in range(4):
        if zoom == None:
            plt.plot(results[i], label = list_label[i])
        else: 
            plt.plot(range(epochs-zoom,epochs),results[i][-zoom:], label = list_label[i])
    if zoom:
        plt.legend(fontsize=7)
    else:
         plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Errors" ) 
    """
    if zoom == None:
        plt.title("Comparing the two framework performances :")
    else:
        plt.title("Comparing the two framework performances zoomed : ")
        
    """
    if file_name != 'no_name':      # Used to store the plots when needed
        plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0)
    

              

def activation_opt_functions_plot(train_error, test_error,epochs , act_opt = False, loss = False, zoom = None, file_name = "no_name" ):
    """
    Display the test and training error across a defined number of epochs 
    
    
    """
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 20
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    if act_opt :
       list_act_opt = ['Vanilla', 'SGD with momentum', 'RMS' , 'Adam']
    elif loss:
        list_act_opt = ['MSELoss', 'BCELoss']
    else:
        list_act_opt = ['Relu', 'Elu' , 'LeakyRelu', 'Tanh', 'Softmax']    
    for i in range(len(train_error)):
        if zoom == None:
            ax[0].plot(train_error[i])
            ax[1].plot(test_error[i])
        else:
            ax[0].plot(range(epochs-zoom,epochs),train_error[i][-zoom:])
            ax[1].plot(range(epochs-zoom,epochs),test_error[i][-zoom:])
        
    ax[0].legend(list_act_opt)
    ax[1].legend(list_act_opt)
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].set_title("Train")
    ax[0].set_ylabel("Errors")
    ax[1].set_ylabel("Errors")
    ax[1].set_title("Test")
    """
    if act_opt:
        fig.suptitle('The  mean error across folds for different optimizers:')
    elif loss:
        fig.suptitle('The  mean error across folds with different criterions:')
    else:
        fig.suptitle('The errors for different activation functions:')
    """   
    if file_name != 'no_name':        # Used to store the plots when needed
        plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()
    

####################################################################################################

def act_opt_functions_plot(train_error, test_error,epochs , act_opt = False, loss = False, zoom = None, file_name = "no_name" ):
    """
    Display the test and training error across a defined number of epochs 
    
    """
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 20
    
    plt.figure(figsize=(8, 6))
    list_color = ['g--', 'g-', 'r--', 'r-','b--', 'b-', 'c--', 'c-']
    if act_opt :
        list_act_opt = ['Vanilla train','Vanilla test', 'Momentum train','Momentum test', 'RMS train', 'RMS test' , 'Adam train', 'Adam test']
    elif loss:
        list_act_opt = ['MSELoss train', 'MSELoss test', 'BCELoss train', 'BCELoss test']
    else:
        list_act_opt = ['Relu train', 'Relu test', 'Elu train','Elu test', 'LeakyRelu train','LeakyRelu test', 'Tanh train', 'Tanh test']    
    j=0
    for i in range(len(train_error)):
        if zoom == None:
            plt.plot(train_error[i],list_color[j])
            plt.plot(test_error[i],list_color[j+1])
            
           
        else:
            plt.plot(range(epochs-zoom,epochs),train_error[i][-zoom:],list_color[j] )
            plt.plot(range(epochs-zoom,epochs),test_error[i][-zoom:],list_color[j+1])
            
        j = j + 2
    if zoom:    
        plt.legend(list_act_opt,fontsize=13)
    else:
        plt.legend(list_act_opt,fontsize=13)
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    
    if file_name != 'no_name':        # Used to store the plots when needed
        plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()
    


####################################################################################################################################



def plot_final_prediction(model, X):
    
    """
    Plot of the data distribution at the end or when needed to be seen :
        
        
    """
    
    X_norm = normalize_data(X)
    output = model.forward(X_norm)
    output[output <0.5] = 0
    output[output > 0.5] = 1
    
    plot_graph(X, output.view(-1), "Testing")
    
 ###############################################################################################   