import os
import numpy as np
from datetime import datetime
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

weight_path = 'weights/'

img_rows = 512
img_cols = 512

## this function is unused now
def depth_softmax(matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

def get_unet():
    inputs = Input((1,img_rows,img_cols))
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

    up1 = merge([UpSampling2D(size=(2, 2))(conv10), conv8], mode='concat', concat_axis=1)
    conv11 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up1)
    conv12 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv11)

    up2 = merge([UpSampling2D(size=(2, 2))(conv12), conv6], mode='concat', concat_axis=1)
    conv13 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up2)
    conv14 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv13)

    up3 = merge([UpSampling2D(size=(2, 2))(conv14), conv4], mode='concat', concat_axis=1)
    conv15 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up3)
    conv16 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv15)

    up4 = merge([UpSampling2D(size=(2, 2))(conv16), conv2], mode='concat', concat_axis=1)
    conv17 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up4)
    conv18 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv17)

    conv19 = Convolution2D(1, 1, 1, activation='sigmoid')(conv18)

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
    model = get_unet()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

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
    model_checkpoint = ModelCheckpoint(os.path.join(dir_path,'weights.{epoch:03d}.hdf5'), monitor='val_acc',save_best_only=True)

    ## Realtime Data Augumentation
    #datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True)
    #datagen.fit(imgs_train_raw)

    model.fit(imgs_train_raw,imgs_train_label,batch_size=2, nb_epoch=5,
        validation_data=[imgs_test_raw,imgs_test_label],callbacks=[model_checkpoint])

    #model.fit_generator(datagen.flow(imgs_test_raw,imgs_test_label,batch_size=2), samples_per_epoch=6, nb_epoch=1,
    #    validation_data=[imgs_test_raw,imgs_test_label],callbacks=[model_checkpoint])

    model.save(os.path.join(weight_path,'unet.hdf5'))

if __name__ == '__main__':
    train()
