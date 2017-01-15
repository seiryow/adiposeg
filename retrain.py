import os
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ios import make_output_dir
from metrics import rand_error_to_patch
from train import batch_size
from predict import predict, clabels_to_img


weight_path = 'weights/'
model_path = 'weights/unet.hdf5'
output_path = 'output/'

nb_epoch = 2


def retrain():
    print '*'*50
    print 'Loading train data...'
    print '*'*50
    imgs_train_raw = np.load('train_raw.npy')
    imgs_train_label = np.load('train_label.npy')

    imgs_test_raw = np.load('test_raw.npy')
    imgs_test_label = np.load('test_label.npy')

    print '*'*50
    print 'Loading the model...'
    print '*'*50
    model = load_model(model_path)

    print '*'*50
    print 'Predict labels on train data...'
    print '*'*50
    pred_clabels = predict(model, imgs_train_raw)

    print '*'*50
    print 'Calculate rand error...'
    print '*'*50
    rand_err = rand_error_to_patch(imgs_train_label, pred_clabels)
    index = np.argsort(rand_err)

    total = index.shape[0]
    new_train_raw = np.zeros([total/2, 1, imgs_train_raw.shape[2], imgs_train_raw.shape[3]], dtype='float32')
    new_train_label = np.zeros([total/2, imgs_train_label.shape[1], 2], dtype='uint8')
    for x in xrange(total/2):
        new_train_raw[x] = imgs_train_raw[index[x]]
        new_train_label[x] = imgs_train_label[index[x]]

    print '*'*50
    print 'Fitting model...'
    print '*'*50
    dir_path = make_output_dir(output_path)

    ## After each epoch if validation_acc is best, save the model
    model_checkpoint = ModelCheckpoint(os.path.join(dir_path, 'weights.{epoch:03d}.hdf5'), monitor='val_acc', save_best_only=True)
    checkpoint2 = ModelCheckpoint(os.path.join(weight_path, 'unet.hdf5'), monitor='val_acc', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')

    ## train
    history = model.fit(new_train_raw, new_train_label, batch_size = batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
            validation_data=[imgs_test_raw, imgs_test_label], callbacks=[model_checkpoint, checkpoint2, early_stopping])

    model.save(os.path.join(dir_path,'result.hdf5'))


if __name__ == '__main__':
    retrain()
