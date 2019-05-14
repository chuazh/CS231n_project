#!/usr/bin/env python

import input_util as ip
import transfer_models as mdl
import vis_util as vis
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import numpy as np

def string_prepend(lr,wt,is_deep):
    
    if is_deep:
        deep_str = 'deep'
    else:
        deep_str = 'shallow'
        
    prepend = 'lr' + str(lr) + '_wt' + str(wt) + '_' + deep_str
    
    return prepend

def train_net(num_epochs,is_deep,num_train_trials,num_val_trials,lr,wt):
    # preliminaries:
    input_size = 224 # for RESNET
    model_name = 'resnet'
    num_output = 14 
    feature_extract = True
    
    # load our dictionary for the samples seq order
    f = open("index_dict.dat",'rb')
    index_dict = pickle.load(f)
    f.close()

    # load the dataset
    train_loader,val_loader,seq_loader = ip.load_dataset(input_size,index_dict,num_train_trials=num_train_trials,num_val_trials=num_val_trials)

    dataloaders_dict = {}
    dataloaders_dict['train']= train_loader
    dataloaders_dict['val'] = val_loader

    #intialize our model
    model_ft, input_size = mdl.initialize_model(model_name, num_output, feature_extract, use_pretrained=True,is_deep = is_deep)
    print(model_ft)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device:')
    print(device)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    #  Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt, amsgrad=False)

    # Setup the loss fxn
    criterion = nn.MSELoss(reduction='mean')

    # Train and evaluate
    model_ft, hist, train_hist = mdl.train_model(model_ft, dataloaders_dict, device, criterion, optimizer_ft, num_epochs=num_epochs,print_every = 5 , is_inception=(model_name=="inception"))
    
    prefix = string_prepend(lr,wt,is_deep)
    
    f = open(prefix+'_model_weights.model','wb')
    torch.save(model_ft.state_dict(), f)
    f.close()

    f = open(prefix+'_train_loss.dat','wb')
    pickle.dump(train_hist,f)
    f.close()

    f = open(prefix+'_val_loss.dat','wb')
    pickle.dump(hist,f)
    f.close()

    vis.check_accuracy_vis(prefix,seq_loader, model_ft, device, plot = False) # don't plot our data
    
    return hist

'''--------------------------------------------------------------------------------------------------'''

num_epochs = 1
num_train_trials = 1
num_val_trials = 1
#lr_list = [1e-3,1e-2,1e-4]
#wt_list = [0,0.01,0.001]
lr_list = [1e-3,1e-2]
wt_list = [0,0.01]
is_deep_list = [False,True]

val_hist_list = []

for is_deep in is_deep_list:
    lr = 0.001
    wt = 0
    val_hist_temp = train_net(num_epochs,is_deep,num_train_trials,num_val_trials,lr,wt)
    val_hist_list.append(val_hist_temp)


dp = np.amin(np.array(val_hist_list[1]))
shal = np.amin(np.array(val_hist_list[0]))

if dp > shal:
    print('shallow is better')
    is_deep = False
else:
    print('deep is better')
    is_deep = True

val_hist_list_final = []
    
for lr in lr_list:
    for wt in wt_list:
        val_hist_temp = train_net(num_epochs,is_deep,num_train_trials,num_val_trials,lr,wt)
        val_hist_list_final.append(val_hist_temp)

