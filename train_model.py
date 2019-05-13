#!/usr/bin/env python

import input_util as ip
import transfer_models as mdl
import vis_util.py as vis
import torch
import torch.optim as optim
import torch.nn as nn

# preliminaries:
input_size = 224 # for RESNET
model_name = 'resnet'
num_output = 14 
feature_extract = True

import pickle

f = open("index_dict.dat",'rb')
index_dict = pickle.load(f)
f.close()

train_loader,val_loader,seq_loader = ip.load_dataset(input_size,index_dict,num_train_trials=30,num_val_trials=5)

dataloaders_dict = {}
dataloaders_dict['train']= train_loader
dataloaders_dict['val'] = val_loader

model_ft, input_size = mdl.initialize_model(model_name, num_output, feature_extract, use_pretrained=True)
print(model_ft)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device:')
print(device)

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
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
optimizer_ft = optim.Adam(params_to_update, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Setup the loss fxn
criterion = nn.MSELoss(reduction='mean')

num_epochs = 1

# Train and evaluate
model_ft, hist, train_hist = mdl.train_model(model_ft, dataloaders_dict, device, criterion, optimizer_ft, num_epochs=num_epochs,print_every = 5 , is_inception=(model_name=="inception"))

f = open("model_weights.model",'wb')
torch.save(model_ft.state_dict(), f)
f.close()

f = open("train_loss.dat",'wb')
index_dict = pickle.dump(train_hist,f)
f.close()

f = open("val_loss.dat",'wb')
index_dict = pickle.dump(hist,f)
f.close()

vis.check_accuracy_vis(loader_seq, model,plot = False) # don't plot our data