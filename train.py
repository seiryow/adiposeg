import os
import numpy as np
import sys
from ios import make_output_dir
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from metrics import rand_error_to_patch
import argparse

weight_path = 'weights/'
#model_path = 'weights/2017-01-10/03-11-13/weights.009.hdf5'

model_load_flag = 0

batch_size = 16 ## batch_size must be smaller than num of samples
nb_epoch = 25


def get_unet(img_rows, img_cols):
    from keras.models import Model
    from keras.layers.core import Reshape, Permute, Activation
    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
    from keras.layers.normalization import BatchNormalization
    from keras import backend as K
    K.set_image_dim_ordering('th')

    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv5)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv8 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv7)
    conv8 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)

    conv9 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
    conv10 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv9)

    deconv1 = Deconvolution2D(512, 2, 2, output_shape=(batch_size, 512, img_rows/8, img_cols/8), subsample=(2, 2), activation='relu')(conv10)
    #deconv1 = UpSampling2D(size=(2,2))(conv10)
    #deconv1 = Convolution2D(512, 2, 2, activation='relu', border_mode='same')(deconv1)
    merge1 = merge([deconv1, conv8], mode='concat', concat_axis=1)
    conv11 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(merge1)
    conv12 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv11)
    conv12 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv12)

    deconv2 = Deconvolution2D(256, 2, 2, output_shape=(batch_size, 256, img_rows/4, img_cols/4), subsample=(2, 2), activation='relu')(conv12)
    #deconv2 = UpSampling2D(size=(2,2))(conv12)
    #deconv2 = Convolution2D(256, 2, 2, activation='relu', border_mode='same')(deconv2)
    merge2 = merge([deconv2, conv6], mode='concat', concat_axis=1)
    conv13 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(merge2)
    conv14 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv13)
    conv14 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv14)

    deconv3 = Deconvolution2D(128, 2, 2, output_shape=(batch_size, 128, img_rows/2, img_cols/2), subsample=(2, 2), activation='relu')(conv14)
    #deconv3 = UpSampling2D(size=(2,2))(conv14)
    #deconv3 = Convolution2D(128, 2, 2, activation='relu', border_mode='same')(deconv3)
    merge3 = merge([deconv3, conv4], mode='concat', concat_axis=1)
    conv15 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(merge3)
    conv16 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv15)

    deconv4 = Deconvolution2D(64, 2, 2, output_shape=(batch_size, 64, img_rows, img_cols), subsample=(2, 2), activation='relu')(conv16)
    #deconv4 = UpSampling2D(size=(2,2))(conv16)
    #deconv4 = Convolution2D(64, 2, 2, activation='relu', border_mode='same')(deconv4)
    merge4 = merge([deconv4, conv2], mode='concat', concat_axis=1)
    conv17 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(merge4)
    conv18 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv17)

    conv19 = Convolution2D(2, 1, 1, activation=None, border_mode='same')(conv18)
    conv19 = Reshape((2,img_rows*img_cols))(conv19)
    conv19 = Permute((2,1))(conv19)
    conv19 = Activation('softmax')(conv19)

    model = Model(input=inputs, output=conv19)
    model.summary()

    return model


def make_history_file(dir_path, hist):
    import csv
    f = open(os.path.join(dir_path,'hist.csv'), 'ab')
    csvWriter = csv.writer(f)

    for key in hist.history:
        tmp = list()
        tmp.append(key)
        for num in hist.history[key]:
            tmp.append(num)
        csvWriter.writerow(tmp)

    f.close()


def train(traindir):
    weight_path = os.path.join(traindir, 'weights')
    # weight_path = traindir + 'weights'
    import shutil
    from keras.utils import plot_model

    print '*'*50
    print 'Loading train data...'
    print '*'*50
    imgs_train_raw = np.load( os.path.join(traindir, 'train_raw.npy'))
    imgs_train_label = np.load( os.path.join(traindir, 'train_label.npy'))

    val_test_raw = np.load( os.path.join(traindir, 'val_test_raw.npy'))
    val_test_label = np.load( os.path.join(traindir, 'val_test_label.npy'))

    # imgs_train_raw = np.load( traindir + 'train_raw.npy')
    # imgs_train_label = np.load( traindir + 'train_label.npy')

    # val_test_raw = np.load( traindir + 'val_test_raw.npy')
    # val_test_label = np.load( traindir + 'val_test_label.npy')


    print '*'*50
    print 'Creating and compiling the model...'
    print '*'*50

    img_rows = imgs_train_raw.shape[2]
    img_cols = imgs_train_raw.shape[3]

    if model_load_flag == 0:
        model = get_unet(img_rows, img_cols)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = load_model(model_path)

    plot_model(model, to_file= os.path.join(traindir, 'model.png'))
    # plot_model(model, to_file= traindir + 'model.png')

    print '*'*50
    print 'Fitting model...'
    print '*'*50

    ## make output dir
    dir_path = make_output_dir(weight_path)

    ## After each epoch if validation_acc is best, save the model

    checkpoint = ModelCheckpoint(os.path.join(weight_path, 'unet.hdf5'),
                                 monitor='val_acc', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')

    ## train
    hist = model.fit(imgs_train_raw, imgs_train_label, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, shuffle=True, validation_data=[val_test_raw, val_test_label],
                     callbacks=[checkpoint, early_stopping])

    shutil.copyfile(os.path.join(weight_path, 'unet.hdf5'), os.path.join(dir_path, 'unet.hdf5'))
    make_history_file(dir_path, hist)

    print 'Done.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess test and train images')
    parser.add_argument('dataset_dir', type=str, help='Directory containing the dataset')
    args = parser.parse_args()
    train(args.dataset_dir)
