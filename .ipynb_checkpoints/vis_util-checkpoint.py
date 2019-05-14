import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np

def check_accuracy_vis(loader, model, device, plot=True):

    print('Checking accuracy on sequential validation set')

    model.eval()  # set model to evaluation mode
    count = 0
    score_array = np.empty((0,14))
    gt_array = np.empty((0,14))
    
    plt.figure()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float)  # move to device, e.g. CPU
            y = y.to(device=device, dtype=torch.float)
            scores = model(x)
            loss_fn = torch.nn.MSELoss(reduction='mean')
            loss = loss_fn(scores,y)
            scores = scores.to(device="cpu",dtype=torch.float)
            y = y.to(device = "cpu", dtype = torch.float)
            if plot:
                plt.plot(range(count, len(scores) + count), scores.numpy()[:,0:3], 'b')
                plt.plot(range(count, len(scores) + count), y.numpy()[:,0:3], 'r')
            
            # append our results
            score_array = np.vstack((score_array,scores.numpy()))
            gt_array = np.vstack((gt_array,y.numpy()))
            
            count = count + len(scores)
        
        #save our results
        print('saving our results...')
        np.savetxt('vis_scores.dat', score_array, delimiter=',')   # X is an array
        np.savetxt('vis_gt.dat', gt_array, delimiter=',')   # X is an array    

        print('MSE loss is: %f ' % loss)
        plt.show()