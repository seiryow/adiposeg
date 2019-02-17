import os
import numpy as np
import sys
from ios import make_output_dir
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from metrics import rand_error_to_patch
import argparse
from pprint import pprint

weight_path = 'weights/'
#model_path = 'weights/2017-01-10/03-11-13/weights.009.hdf5'

model_load_flag = 0

batch_size = 16 ## batch_size must be smaller than num of samples
nb_epoch = 25


def get_unet(img_rows, img_cols):
    from keras.models import Model
    from keras.layers.core import Reshape, Permute, Activation
    from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Deconvolution2D, Add
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import concatenate

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv6)

    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv8)

    conv9 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv10 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv9)

    deconv1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation="relu", padding='same')(conv10)
    merge1 = concatenate([deconv1, conv8], axis=3)
    conv11 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge1)
    conv12 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv11)
    conv12 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv12)

    deconv2 = Conv2DTranspose(256, (2, 2), activation="relu", strides=(2, 2), padding='same')(conv12)
    merge2 = concatenate([deconv2, conv6], axis=3)
    conv13 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge2)
    conv14 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv13)
    conv14 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv14)

    deconv3 = Conv2DTranspose(128, (2, 2), activation="relu", strides=(2, 2), padding='same')(conv14)
    merge3 = concatenate([deconv3, conv4], axis=3)
    conv15 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge3)
    conv16 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv15)

    deconv4 = Conv2DTranspose(64, (2, 2), activation="relu", strides=(2, 2), padding='same')(conv16)
    merge4 = concatenate([deconv4, conv2], axis=3)
    conv17 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge4)
    conv18 = Conv2D(64, (2, 2), activation='relu', padding='same')(conv17)

    conv19 = Conv2D(2, (1, 1), activation=None, padding='same')(conv18)
    conv19 = Reshape((img_rows*img_cols, 2))(conv19)
    conv19 = Activation('softmax')(conv19)
    
    model = Model(inputs=inputs, outputs=conv19)
    model.summary()

    return model


def make_history_file(dir_path, hist):
    import csv
    pprint(dir_path)
    pprint(hist)
    f = open(os.path.join(dir_path,'hist.csv'), 'a')
    csvWriter = csv.writer(f)

    for key in hist.history:
        pprint(key)
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

    print('*'*50)
    print('Loading train data...')
    print('*'*50)
    imgs_train_raw = np.load( os.path.join(traindir, 'train_raw.npy'))
    imgs_train_label = np.load( os.path.join(traindir, 'train_label.npy'))

    val_test_raw = np.load( os.path.join(traindir, 'val_test_raw.npy'))
    val_test_label = np.load( os.path.join(traindir, 'val_test_label.npy'))

    # imgs_train_raw = np.load( traindir + 'train_raw.npy')
    # imgs_train_label = np.load( traindir + 'train_label.npy')

    # val_test_raw = np.load( traindir + 'val_test_raw.npy')
    # val_test_label = np.load( traindir + 'val_test_label.npy')


    print('*'*50)
    print('Creating and compiling the model...')
    print('*'*50)
    pprint(imgs_train_raw.shape)
    img_rows = imgs_train_raw.shape[1]
    img_cols = imgs_train_raw.shape[2]

    if model_load_flag == 0:
        model = get_unet(img_rows, img_cols)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = load_model(model_path)

    plot_model(model, to_file= os.path.join(traindir, 'model.png'))
    # plot_model(model, to_file= traindir + 'model.png')

    print('*'*50)
    print('Fitting model...')
    print('*'*50)

    ## make output dir
    dir_path = make_output_dir(weight_path)
    print("weight_path")
    pprint(weight_path)
    print("dir_path")
    pprint(dir_path)
    ## After each epoch if validation_acc is best, save the model

    checkpoint = ModelCheckpoint(os.path.join(dir_path, 'unet_model.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='val_acc', save_best_only=False, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')

    ## train
    hist = model.fit(imgs_train_raw, imgs_train_label, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, shuffle=True, validation_data=[val_test_raw, val_test_label],
                     callbacks=[checkpoint, early_stopping])

    shutil.copyfile(os.path.join(dir_path, 'unet.hdf5'), os.path.join(weight_path, 'unet.hdf5'))
    make_history_file(dir_path, hist)

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess test and train images')
    parser.add_argument('dataset_dir', type=str, help='Directory containing the dataset')
    args = parser.parse_args()
    train(args.dataset_dir)
