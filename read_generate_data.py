#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:23:18 2019

@author: bran
"""

from __future__ import print_function
#import skimage
import scipy
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage


#train_list = ['4', '5', '7', '070', '14']
train_list = ['1','4', '5', '070']
#val_list = ['7', '14' '148']
val_list = ['148']
stride = [4, 16, 16]
patch_size = [8, 32, 32]
seg_path = 'training'
cut = 10
thresh = 10

train_data = []
train_gt = []

val_data = []
val_gt = []

dirs = os.listdir(seg_path)
dirs.sort()
count = 0

def generate_patches(in_array, stride, patch_size):
    out_array = []
    for ii in range(0, np.shape(in_array)[0]-patch_size[0], stride[0]):
        for jj in range(0, np.shape(in_array)[1]-patch_size[1], stride[1]):
            for kk in range(0, np.shape(in_array)[2]-patch_size[2], stride[2]):            
                patch = in_array[ii:ii+patch_size[0], jj:jj+patch_size[1], kk:kk+patch_size[2]]
                out_array.append(patch)
    return out_array
  #  return np.asarray(out_array)

for dir_name in train_list:
   print(dir_name)
   seg_img = sitk.ReadImage(os.path.join(seg_path, dir_name, 'segm.nii.gz'))
   seg_array = sitk.GetArrayFromImage(seg_img)
   seg_array = seg_array[:, cut:np.shape(seg_array)[1]-cut, cut:np.shape(seg_array)[2]-cut]

   flair_img = sitk.ReadImage(os.path.join(seg_path, dir_name, 'pre','FLAIR.nii.gz'))
   flair_array = sitk.GetArrayFromImage(flair_img)

   brain_mask_flair = np.zeros(np.shape(flair_array), dtype = 'float32')
   brain_mask_flair[flair_array >=thresh] = 1
   brain_mask_flair[flair_array < thresh] = 0
   for iii in range(np.shape(flair_array)[0]):
       brain_mask_flair[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_flair[iii,:,:])  #fill the holes inside br
   flair_array = flair_array - np.mean(flair_array[brain_mask_flair == 1])
   flair_array /= np.std(flair_array[brain_mask_flair == 1])
   flair_array = flair_array[:, cut:np.shape(flair_array)[1]-cut, cut:np.shape(flair_array)[2]-cut]

   t1_img = sitk.ReadImage(os.path.join(seg_path, dir_name, 'pre','reg_T1.nii.gz'))
   t1_array = sitk.GetArrayFromImage(t1_img)

   brain_mask_t1 = np.zeros(np.shape(t1_array), dtype = 'float32')
   brain_mask_t1[t1_array >=thresh] = 1
   brain_mask_t1[t1_array < thresh] = 0
   for iii in range(np.shape(t1_array)[0]):
       brain_mask_t1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_t1[iii,:,:])  #fill the holes inside br
   t1_array = t1_array - np.mean(t1_array[brain_mask_t1 == 1])
   t1_array /= np.std(t1_array[brain_mask_t1 == 1])
   t1_array = t1_array[:, cut:np.shape(t1_array)[1]-cut, cut:np.shape(t1_array)[2]-cut]

   patch_seg = generate_patches(seg_array, stride, patch_size)
   patch_flair = generate_patches(flair_array, stride, patch_size)
   patch_t1 = generate_patches(t1_array, stride, patch_size)

   patch_seg_ = np.concatenate([arr[np.newaxis] for arr in patch_seg])
   patch_seg_ = patch_seg_[..., np.newaxis]
   patch_flair_ = np.concatenate([arr[np.newaxis] for arr in patch_flair])
   patch_t1_ = np.concatenate([arr[np.newaxis] for arr in patch_t1])
   patch_flair_t1 = np.concatenate((patch_flair_[..., np.newaxis], patch_t1_[..., np.newaxis]), axis = 4)

   train_data.extend(patch_flair_t1)
   train_gt.extend(patch_seg_)

train_data_ = np.concatenate([arr[np.newaxis] for arr in train_data])
train_gt_ = np.concatenate([arr[np.newaxis] for arr in train_gt])
np.save('train_data.npy', train_data_)
np.save('train_gt.npy', train_gt_)


for dir_name in val_list:   
    print(dir_name)
    seg_img = sitk.ReadImage(os.path.join(seg_path, dir_name, 'segm.nii.gz'))
    seg_array = sitk.GetArrayFromImage(seg_img)
    seg_array = seg_array[:, cut:np.shape(seg_array)[1]-cut, cut:np.shape(seg_array)[2]-cut]
    
    flair_img = sitk.ReadImage(os.path.join(seg_path, dir_name, 'pre','FLAIR.nii.gz'))
    flair_array = sitk.GetArrayFromImage(flair_img)
    
    brain_mask_flair = np.zeros(np.shape(flair_array), dtype = 'float32')
    brain_mask_flair[flair_array >=thresh] = 1
    brain_mask_flair[flair_array < thresh] = 0
    for iii in range(np.shape(flair_array)[0]):
        brain_mask_flair[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_flair[iii,:,:])  #fill the holes inside br
    flair_array = flair_array - np.mean(flair_array[brain_mask_flair == 1])
    flair_array /= np.std(flair_array[brain_mask_flair == 1])
    flair_array = flair_array[:, cut:np.shape(flair_array)[1]-cut, cut:np.shape(flair_array)[2]-cut]
    
    t1_img = sitk.ReadImage(os.path.join(seg_path, dir_name, 'pre','reg_T1.nii.gz'))
    t1_array = sitk.GetArrayFromImage(t1_img)
    
    brain_mask_t1 = np.zeros(np.shape(t1_array), dtype = 'float32')
    brain_mask_t1[t1_array >=thresh] = 1
    brain_mask_t1[t1_array < thresh] = 0
    for iii in range(np.shape(t1_array)[0]):
        brain_mask_t1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_t1[iii,:,:])  #fill the holes inside br
    t1_array = t1_array - np.mean(t1_array[brain_mask_t1 == 1])
    t1_array /= np.std(t1_array[brain_mask_t1 == 1])
    t1_array = t1_array[:, cut:np.shape(t1_array)[1]-cut, cut:np.shape(t1_array)[2]-cut]
    
    
    seg_array = seg_array[...,np.newaxis]
    flair_t1 = np.concatenate((flair_array[...,np.newaxis], t1_array[..., np.newaxis]), axis = 3)
    flair_t1 = flair_t1[..., np.newaxis]
    
    np.save('val_'+dir_name+'_data.npy', flair_t1)
    np.save('val_'+dir_name+'_gt.npy', seg_array)
