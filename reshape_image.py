#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:59:29 2019

@author: bran
"""

from __future__ import print_function
from __future__ import division

# import click
import json
import os
import nibabel as nib
import numpy as np
from models.threeDUNet import get_3Dunet
import SimpleITK as sitk

from eval.evaluation import getDSC, getHausdorff, getVS
from metrics import dice_coef, dice_coef_loss



def main():

    img148 = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage("results/result_148.nii.gz")),[1,2,0])
    img1 = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage("results/result_1.nii.gz")),[1,2,0])
    img148[img148 == 9] = 0
    img1[img1 == 9] = 0


    img148_new = np.pad(img148, [(10, 10), (10, 10), (0,0)], mode='constant', constant_values=0)
    img1_new = np.pad(img1, [(10, 10), (10, 10), (0,0)], mode='constant', constant_values=0)

    #img148_new = nib.Nifti1Image(img148_new, None)
    #img148_new.to_filename("results2/result_148.nii.gz")
    nib.save(nib.Nifti1Image(img148_new, None), os.path.join("results2/", "result_148.nii.gz"))
    nib.save(nib.Nifti1Image(img1_new, None), os.path.join("results2/", "result_1.nii.gz"))



if __name__ == '__main__':
    main()