import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler, Dataset
import os
import numpy
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models
import glob
from PIL import Image
import cv2
import numpy as np
import pickle
from natsort import humansorted
import pdb

def collect_video_framecount(action,subject,trial_num):
    '''
    This function takes the video and splits it into frames
    Inputs:
    Outputs:
    '''
    action_dict = {'KT':'Knot_Tying','S':'Suturing','NP': 'Needle_Passing'}
    
    act = action_dict[action]

    filename1 = act+'/video/'+act+'_'+subject+'00'+str(trial_num)+'_capture1.avi'
    filename2 = act+'/video/'+act+'_'+subject+'00'+str(trial_num)+'_capture2.avi'
    
    print('reading '+filename1)
    
    vidcap1 = cv2.VideoCapture(filename1)
    vidcap2 = cv2.VideoCapture(filename2)
    
    count = 0
    success = True
    
    while success:
      success,image = vidcap1.read()
    
      '''
      success,image = vidcap2.read()
      '''  
      count += 1

    print('total frame count : %d' % count)
    
    return count-1

def collect_video_sample(action,subject,trial_num,num_frames):
    
    action_dict = {'KT':'Knot_Tying','S':'Suturing','NP': 'Needle_Passing'}
    
    act = action_dict[action]

    filename1 = act+'/video/'+act+'_'+subject+'00'+str(trial_num)+'_capture1.avi'
    filename2 = act+'/video/'+act+'_'+subject+'00'+str(trial_num)+'_capture2.avi'
    
    print('reading '+filename1)
    
    vidcap1 = cv2.VideoCapture(filename1)
    vidcap2 = cv2.VideoCapture(filename2)
    
    # collect kinematic data
    filepath = act + '/kinematics/AllGestures/'
    filename = filepath + act + '_' +subject + '00' + str(trial_num) + '.txt'
    data = np.loadtxt(filename)
    num_labels = data.shape[0]
    print('total labels loaded: %d' % num_labels)
    
    if (num_labels>num_frames):
          pass
    else:
          num_frames = num_labels
          
    count = 0
    success = True
    
    while success and count<num_frames:
      success,image = vidcap1.read()
      write_name = 'data/' + subject+'_'+str(trial_num)+'_1'+'_%d_'+ action + '.png'
      cv2.imwrite(write_name % count, image)     # save frame as png file
    
      '''
      success,image = vidcap2.read()
      write_name = 'data/' + subject+'_'+str(trial_num)+'_2'+'_%d_'+ action + '.png'
      cv2.imwrite(write_name % count, image)     # save frame as png file
      '''
      count += 1
      if count%100 == 0:
          print('capturing frame %d' % count)  
    
    print('total frame count : %d' % count)
    
    # only take cols 38-49 (slave left) and 57-68 (slave right)
    
    dataL = data[:count,38:50]
    dataR = data[:count,57:69]
    
    out = np.hstack((dataL,dataR))
    
    print('total labels saved: %d' % out.shape[0])
    
    return out

class JIGSAWDataset(Dataset):

    def __init__(self, y, sortedFilelist , transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            sortedFilelist (string): sorted list of filenames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = y
        self.sortedlist = sortedFilelist
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]
        
    def __getitem__(self, idx): 
        img_name = self.sortedlist[idx]
        #print('opening image ' +  img_name)
        image = Image.open(img_name,'r')
        label = self.labels[idx,:]
        if self.transform:
            image = self.transform(image)
        sample =  (image,label)

        return sample

def load_dataset(input_size):
    
    data_path = 'data'
    picklefile = open("kinematics", "rb" )
    
    num_files = len(next(os.walk('data'))[2]) #dir is your directory path as string
    print('num image files: %d' % num_files)
    
    trans = T.Compose([
                T.CenterCrop(240),
                T.Resize((input_size), interpolation=2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transy = T.Compose([T.ToTensor()])

    y = pickle.load(picklefile)
    print('shape of label data: %dx%d' % (y.shape[0],y.shape[1]))
    picklefile.close()
    
    file_list = glob.glob('data/*.png')
    sortedlist =  humansorted(file_list)
    
    dataset = JIGSAWDataset(y,sortedlist,transform = trans)
    
    train_loader = DataLoader(
        dataset,
        batch_size=25,
        num_workers=0,
        shuffle=False,
        sampler=sampler.SubsetRandomSampler(range(num_files-2000))
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=25,
        num_workers=0,
        shuffle=False,
        sampler=sampler.SubsetRandomSampler(range(num_files-2000,num_files))
    )
    
    return train_loader,val_loader
    
