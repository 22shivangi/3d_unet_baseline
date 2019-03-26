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
        #print(ii)
        for jj in range(0, np.shape(test_array)[1]-patch_size[1], stride[1]):
            for kk in range(0, np.shape(test_array)[2]-patch_size[2], stride[2]):
                patch_output = output[:, ii:ii+patch_size[0], jj:jj+patch_size[1], kk:kk+patch_size[2],:]
                input_patch = test_array[np.newaxis, ii:ii+patch_size[0], jj:jj+patch_size[1], kk:kk+patch_size[2], :, 0]
                pred_patch =  model.predict(input_patch)
                patch_output = patch_output+pred_patch
                output[:, ii:ii+patch_size[0], jj:jj+patch_size[1], kk:kk+patch_size[2],:] = patch_output
    return output


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_classes = 10
    # learn_rate = 1e-5
    stride = [4, 16, 16]
    test_imgs_7 = np.load('val_data.npy')
    test_masks_7 = np.load('val_gt.npy')
    test_imgs14 = np.load('test_14_data.npy')
    test_masks14 = np.load('test_14_gt.npy')

    img_shape = (8, 32, 32, 2)
    patch_size = (8, 32, 32)
    model = get_3Dunet(img_shape=img_shape, num_classes=num_classes)
#    assert os.path.isfile(pretrained_model)
    model.load_weights('weights/initial_74.h5')


    pred_masks14 = global_prediction(model, test_imgs14, patch_size, stride)
    pred_masks14 = pred_masks14.argmax(axis=4)
    print(pred_masks14.shape)
    print(np.transpose(pred_masks14[0, :, :, :],).shape)

    save_image(image=np.transpose(pred_masks14[0, :, :, :],), img_num="14")
    pred_masks14 = pred_masks14[0, :, :, :, np.newaxis]

    dsc, h95, vs = get_eval_metrics(test_masks14[...,0], pred_masks14[...,0])
    print(len(test_masks14[...,0]))
    print(test_masks14[...,0])
    print("Subject 14")
    print(dsc)
    print(h95)
    print(vs)



    pred_masks7 = global_prediction(model, test_imgs_7, patch_size, stride)
    pred_masks7 = pred_masks7.argmax(axis=4)

    save_image(image=np.transpose(pred_masks7[0, :, :, :],), img_num="7")
    pred_masks7 = pred_masks7[0, :, :, :, np.newaxis]

    dsc, h95, vs = get_eval_metrics(test_masks_7[..., 0], pred_masks7[..., 0])
    print("Subject 7")
    print(dsc)
    print(h95)
    print(vs)


def save_image(image,img_num="1",direc="results/"):
    img = nib.Nifti1Image(image, None)
    imgname = 'result_' + str(img_num) + '.nii.gz'
    nib.save(img, os.path.join(direc, imgname))



if __name__ == '__main__':
    main()