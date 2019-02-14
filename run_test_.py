#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:59:29 2019

@author: bran
"""

from __future__ import print_function
from __future__ import division

#import click
import json
import os
import nibabel as nib
import numpy as np
from models.threeDUNet import get_3Dunet
import SimpleITK as sitk

from eval.evaluation import getDSC, getHausdorff, getVS
from metrics import dice_coef, dice_coef_loss


def get_eval_metrics(true_mask, pred_mask):
    true_mask_sitk = sitk.GetImageFromArray(true_mask)
    pred_mask_sitk = sitk.GetImageFromArray(pred_mask)
    dsc = getDSC(true_mask_sitk, pred_mask_sitk)
    h95 = getHausdorff(true_mask_sitk, pred_mask_sitk)
    vs = getVS(true_mask_sitk, pred_mask_sitk)

    result = {}
    result['dsc'] = dsc
    result['h95'] = h95
    result['vs'] = vs

    return (dsc, h95, vs)


#@click.command()
#@click.argument('test_imgs_np_file', type=click.STRING)
#@click.argument('test_masks_np_file', type=click.STRING)
#@click.argument('pretrained_model', type=click.STRING)
#@click.option('--output_pred_mask_file', type=click.STRING, default='')
#@click.option('--output_metric_file', type=click.STRING, default='')
    
def global_prediction(model, test_array, patch_size, stride):
    output = np.zeros((1, np.shape(test_array)[0], np.shape(test_array)[1], np.shape(test_array)[2], 10), dtype = 'float32')
    for ii in range(0, np.shape(test_array)[0]-patch_size[0], stride[0]):
        print(ii)
        for jj in range(0, np.shape(test_array)[1]-patch_size[1], stride[1]):
            for kk in range(0, np.shape(test_array)[2]-patch_size[2], stride[2]):
                patch_output = output[:, ii:ii+patch_size[0], jj:jj+patch_size[1], kk:kk+patch_size[2],:]
                input_patch = test_array[np.newaxis, ii:ii+patch_size[0], jj:jj+patch_size[1], kk:kk+patch_size[2], :, 0]
                pred_patch =  model.predict(input_patch)
                patch_output = patch_output+pred_patch
                output[:, ii:ii+patch_size[0], jj:jj+patch_size[1], kk:kk+patch_size[2],:] = patch_output
    return output


def main():
    num_classes = 10
    # learn_rate = 1e-5
    stride = [4, 16, 16]
    test_imgs_1 = np.load('val_1_data.npy')
    test_masks_1 = np.load('val_1_gt.npy')
    test_imgs148 = np.load('val_148_data.npy')
    test_masks148 = np.load('val_148_gt.npy')

    img_shape = (8, 32, 32, 2)
    patch_size = (8, 32, 32)
    model = get_3Dunet(img_shape=img_shape, num_classes=num_classes)
#    assert os.path.isfile(pretrained_model)
    model.load_weights('weights/best_baseline.h5')


    pred_masks148 = global_prediction(model, test_imgs148, patch_size, stride)
    pred_masks148 = pred_masks148.argmax(axis=4)

    save_image(image=np.transpose(pred_masks148[0, :, :, :],[1,2,0]), img_num="148")
    pred_masks148 = pred_masks148[0, :, :, :, np.newaxis]

    np.save('pred_masks148.npy', pred_masks148)
    dsc, h95, vs = get_eval_metrics(test_masks148[...,0], pred_masks148[...,0])
    print("Subject 148")
    print(dsc)
    print(h95)
    print(vs)



    pred_masks1 = global_prediction(model, test_imgs_1, patch_size, stride)
    pred_masks1 = pred_masks1.argmax(axis=4)
    pred_masks1 = pred_masks1[0, :, :, :, np.newaxis]
    save_image(image=pred_masks1, img_num="1")
    np.save('pred_masks1.npy', pred_masks1)
    dsc, h95, vs = get_eval_metrics(test_masks_1[..., 0], pred_masks1[..., 0])
    print("Subject 1")
    print(dsc)
    print(h95)
    print(vs)


def save_image(image,img_num="1",direc="results/"):
    img = nib.Nifti1Image(image, None)
    imgname = 'result_' + str(img_num) + '.nii.gz'
    nib.save(img, os.path.join(direc, imgname))



if __name__ == '__main__':
    main()