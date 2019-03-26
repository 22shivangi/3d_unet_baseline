from __future__ import print_function
from __future__ import division

#import click
import json
import os
import math
import numpy as np
from sklearn.utils import class_weight
from tensorflow import keras
Adam = keras.optimizers.Adam
to_categorical = keras.utils.to_categorical
from models.threeDUNet import get_3Dunet
from tf_logging.tf_logger import Logger

from metrics import weighted_categorical_crossentropy

def main(eval_per_epoch = True, use_augmentation=False, use_weighted_crossentropy=False):
#    assert (test_imgs_np_file != '' and test_masks_np_file != '') or \
#           (test_imgs_np_file == '' and test_masks_np_file == ''), \
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    logger = Logger(model_name='gan', data_name="mr_brains", log_path="tf_logs/")
    num_classes = 10
    if not use_augmentation:
        total_epochs = 800
    else:
        total_epochs = 500
    batch_size = 32
    lr = 1e-4
    # if eval_per_epoch:
    #     test_imgs_1 = np.load('val_1_data.npy')
    #     test_masks_1 = np.load('val_1_gt.npy')
    #     test_imgs_2 = np.load('val_148_data.npy')
    #     test_masks_2 = np.load('val_148_gt.npy')
        
    train_imgs = np.load('train_data.npy')
    train_masks = np.load('train_gt.npy')
    val_imgs = np.load('val_data.npy')
    val_masks = np.load('val_gt.npy')

    if use_weighted_crossentropy:
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_masks),
                                                          train_masks.flatten())
    img_shape = (train_imgs.shape[1], train_imgs.shape[2], train_imgs.shape[3], 2)
    model = get_3Dunet(img_shape=img_shape, num_classes=num_classes)
#    if pretrained_model != '':
#        assert os.path.isfile(pretrained_model)
#        model.load_weights(pretrained_model)

    train_masks_cat = to_categorical(train_masks, num_classes)
    val_masks_cat = to_categorical(val_masks, num_classes)

    if use_weighted_crossentropy:
        model.compile(optimizer=Adam(lr=(lr), decay=0.01), loss=weighted_categorical_crossentropy(class_weights), metrics=['acc'])
    else:
        model.compile(optimizer=Adam(lr=(lr), decay=0.01), loss='categorical_crossentropy', metrics=['acc'])
#    model.fit(train_imgs, train_masks_cat, batch_size=batch_size, epochs=total_epochs, verbose=True, shuffle=True)
    current_epoch = 1
    history = {}
    history['dsc'] = []
    history['h95'] = []
    history['vs'] = []

    best_val_acc = 0
    best_val_loss = np.inf
    while current_epoch <= total_epochs:
        print('Epoch', str(current_epoch), '/', str(total_epochs))
        hist = model.fit(x=train_imgs, y=train_masks_cat, batch_size=batch_size, epochs=1, verbose=True, shuffle=True,
                               validation_data=(val_imgs,val_masks_cat))

        logger.log_loss(mode='train', loss=hist.history['loss'][0], epoch=current_epoch)
        logger.log_loss(mode='val', loss=hist.history['val_loss'][0], epoch=current_epoch)
        logger.log_acc(mode='train', acc=hist.history['acc'][0], epoch=current_epoch)
        logger.log_acc(mode='val', acc=hist.history['val_acc'][0], epoch=current_epoch)

        if best_val_acc < hist.history['val_acc'][0] and best_val_loss > hist.history['val_loss'][0]:
            best_val_acc = hist.history['val_acc'][0]
            best_val_loss = hist.history['val_loss'][0]
            model.save_weights('weights/initial_' + str(current_epoch) + '.h5')

#        if eval_per_epoch and current_epoch % 100 == 0:
#            model.save_weights(output_weights_file)
#            pred_masks = model.predict(test_imgs)
#            pred_masks = pred_masks.argmax(axis=3)
#            dsc, h95, vs = get_eval_metrics(test_masks[:, :, :, 0], pred_masks)
#            history['dsc'].append(dsc)
#            history['h95'].append(h95)
#            history['vs'].append(vs)
#            print(dsc)
#            print(h95)
#            print(vs)
#            if output_test_eval != '':
#                with open(output_test_eval, 'w+') as outfile:
#                    json.dump(history, outfile)

        current_epoch += 1

    #model.save_weights(output_weights_file)

#    if output_test_eval != '':
#        with open(output_test_eval, 'w+') as outfile:
#            json.dump(history, outfile)



if __name__ == "__main__":
    main()