import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler, Dataset
import os
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
import random

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

def collect_video_sample(action,subject,trial_num):
    
    num_frames = collect_video_framecount(action,subject,trial_num)
    
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
    
    return out

def collect_labels(act,task,subj,trial):
    
    y = np.empty((0,14)) # 6 for position, 24 for direction cosines, 14 for quaternion

    index_dict = {}

    for i in range(len(subj)):
        for j in trial[i]:
            # collect kinematic data
            count = collect_video_framecount(task,subj[i],j+1)
            filepath = act + '/kinematics/AllGestures/'
            filename = filepath + act + '_' + subj[i] + '00' + str(j+1) + '.txt'
            data = np.loadtxt(filename)

            r_test = data[0, 3:12]

            num_labels = data.shape[0]
            print('total labels loaded: %d' % num_labels)
            dataL_pos = data[:count,39:42]
            dataR_pos = data[:count,57:60]

            dataL_rot = data[:count, 41:50]
            dataR_rot = data[:count, 60:69]

            # now we change the representation of the rotation to quaternion
            N = dataL_rot.shape[0]
            dataL_quat = np.zeros((N, 4))
            dataR_quat = np.zeros((N, 4))
            for k in range(N):
                L_rot = np.asarray(dataL_rot[k, :]).reshape((3,3))
                R_rot = np.asarray(dataR_rot[k, :]).reshape((3,3))

                dataL_quat[k,:] = quaternion_from_matrix(L_rot)
                dataR_quat[k,:] = quaternion_from_matrix(R_rot)

            dataL = np.hstack((dataL_pos, dataL_quat))
            dataR = np.hstack((dataR_pos, dataR_quat))
            out = np.hstack((dataL,dataR))
            print(out.shape)

            index_dict[subj[i] + '00' + str(j+1)] = list(range(y.shape[0], y.shape[0] + out.shape[0]))
            y = np.vstack((y,out))
            
    return y, index_dict
    
    
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
        return self.labels.shape[0]
        
    def __getitem__(self, idx): 
        img_name = self.sortedlist[idx]
        #print('opening image ' +  img_name)
        image = Image.open(img_name,'r')
        label = self.labels[idx,:]
        if self.transform:
            image = self.transform(image)
        sample =  (image,label)

        return sample

def load_dataset(input_size,index_dict,num_train_trials=3,num_val_trials=3):
    
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
    range_total = random.sample(list(index_dict.values()), k=(num_train_trials + num_val_trials))
    range_train = [range_total[i] for i in range(num_train_trials)]
    range_train = sum(range_train,[])
    range_val = [range_total[i] for i in range(num_train_trials,num_train_trials+num_val_trials)]
    range_val = sum(range_val,[])
    sortedlist_seq = [ sortedlist[i] for i in range_val]
    seq_dataset = JIGSAWDataset(y[range_val,:],sortedlist_seq,transform = trans)
    
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        sampler=sampler.SubsetRandomSampler(range_train)
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        sampler=sampler.SubsetRandomSampler(range_val)
    )
    
    seq_loader = DataLoader(
        seq_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        sampler=sampler.SequentialSampler(seq_dataset)
    )
    
    return train_loader,val_loader, seq_loader
    
# GOT THIS ONLINE, NEED TO FIGURE OUT HOW TO CITE IT https://www.lfd.uci.edu/~gohlke/code/transformations.py.html

def quaternion_from_matrix(matrix, isprecise=False):

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q