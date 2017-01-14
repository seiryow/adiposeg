import os
import numpy as np
import keras
from datetime import datetime
from keras.models import Model, load_model
from keras.layers.core import Reshape, Permute, Activation
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

weight_path = 'weights/'
model_path = 'weights/2017-01-10/03-11-13/weights.009.hdf5'

model_load_flag = 0

img_rows = 128
img_cols = 128

batch_size = 16
nb_epoch = 20

def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv8 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv7)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)

    conv9 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
    conv10 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv9)

    deconv1 = Deconvolution2D(512, 2, 2, output_shape=(batch_size, 512, img_rows/8, img_cols/8), subsample=(2, 2), activation='relu')(conv10)
    #deconv1 = UpSampling2D(size=(2,2))(conv10)
    #deconv1 = Convolution2D(512, 2, 2, activation='relu', border_mode='same')(deconv1)
    merge1 = merge([deconv1, conv8], mode='concat', concat_axis=1)
    conv11 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(merge1)
    conv12 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv11)

    deconv2 = Deconvolution2D(256, 2, 2, output_shape=(batch_size, 256, img_rows/4, img_cols/4), subsample=(2, 2), activation='relu')(conv12)
    #deconv2 = UpSampling2D(size=(2,2))(conv12)
    #deconv2 = Convolution2D(256, 2, 2, activation='relu', border_mode='same')(deconv2)
    merge2 = merge([deconv2, conv6], mode='concat', concat_axis=1)
    conv13 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(merge2)
    conv14 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv13)

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

def train():
    print '-'*30
    print 'Loading train data...'
    print '-'*30
    imgs_train_raw = np.load('train_raw.npy')
    imgs_train_label = np.load('train_label.npy')

    imgs_test_raw = np.load('test_raw.npy')
    imgs_test_label = np.load('test_label.npy')

    print '-'*30
    print 'Creating and compiling model...'
    print '-'*30

    if model_load_flag == 0:
        model = get_unet()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = load_model(model_path)

    ##make output dir
    time = datetime.now()

    day_dir = time.strftime('%Y-%m-%d')
    dir_path = os.path.join(weight_path,day_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    time_dir = time.strftime('%H-%M-%S')
    dir_path = os.path.join(dir_path,time_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    print '-'*30
    print 'Fitting model...'
    print '-'*30

    ## After each epoch if validation_acc is best, save the model
    model_checkpoint = ModelCheckpoint(os.path.join(dir_path, 'weights.{epoch:03d}.hdf5'), monitor='val_acc', save_best_only=True)

    ## train
    history = model.fit(imgs_train_raw, imgs_train_label, batch_size = batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
            validation_data=[imgs_test_raw, imgs_test_label], callbacks=[model_checkpoint])

    model.save(os.path.join(dir_path,'result.hdf5'))
    model.save(os.path.join(weight_path,'unet.hdf5'))

    print 'Done.'

if __name__ == '__main__':
    train()
