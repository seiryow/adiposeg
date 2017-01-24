import os
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ios import make_output_dir
from metrics import rand_error_to_img, rand_error_to_patch
from predict import binary_predict, clabels_to_img


weight_path = 'weights/'
#model_path = 'weights/unet.hdf5'
model_path = 'weights/2017-01-13/19-53-09/weights.005.hdf5'
output_path = 'output/'

nb_epoch = 20


def make_A_array(imgs_train_raw, imgs_train_label, train_pred):
    from preprocess import categorize_label
    from image import img_to_array, array_to_img, flip_img, rotate_img

    rand_err = rand_error_to_patch(imgs_train_label, train_pred)
    sort = np.sort(rand_err) # if sort [::-1]
    index = np.argsort(sort)

    total = index.shape[0] / 100 * 25
    if total % 16 != 0:
        total += 16 - (total % 16)              #for deconv
    print 'total:', total

    a_train_raw = np.zeros([total, 1, imgs_train_raw.shape[2], imgs_train_raw.shape[3]], dtype='float32')
    a_train_label = np.zeros([total, train_pred.shape[1], 2], dtype='uint8')

    i = 0
    for x in xrange(total):
        a_train_raw[x] = imgs_train_raw[index[x]]
        a_train_label[x] = imgs_train_label[index[x]]

    return a_train_raw, a_train_label


def make_B_array(imgs_test_raw, imgs_test_label, test_pred):
    import math
    from image import img_to_array, array_to_img, flip_img, rotate_img

    test_total = test_pred.shape[0]
    prob = np.zeros([test_total], dtype="float32")
    for x in xrange(test_total):
        for i in xrange(test_pred.shape[1]):
            prob[x] = math.fabs(test_pred[x][i][1] - 0.5)

    sort = np.sort(prob)[::-1] # if sort [::-1] discent
    index = np.argsort(sort)

    btotal = index.shape[0] / 100 * 25
    if btotal % 16 != 0:
        btotal += 16 - (btotal % 16)              #for deconv
    print 'btotal:', btotal

    b_train_raw = np.zeros([btotal, 1, imgs_test_raw.shape[2], imgs_test_raw.shape[3]], dtype='float32')
    b_train_label = np.zeros([btotal, test_pred.shape[1], 2], dtype='uint8')

    i = 0
    for x in xrange(btotal):
        b_train_raw[x] = imgs_test_raw[index[x]]
        b_train_label[x] = np.round(test_pred[index[x]])

    return b_train_raw, b_train_label


def retrain():
    print '*'*50
    print 'Loading train data...'
    print '*'*50
    imgs_train_raw = np.load('retrain_raw.npy')
    imgs_train_label = np.load('retrain_label.npy')

    imgs_test_raw = np.load('test_raw.npy')
    imgs_test_label = np.load('test_label.npy')

    dir_path = make_output_dir(weight_path)

    prev_path = os.path.join(dir_path, 'prev.hdf5')
    current_path = os.path.join(dir_path, 'current.hdf5')

    import shutil
    shutil.copyfile(model_path, prev_path)
    shutil.copyfile(model_path, current_path)

    k = 5
    prev_rand = 1
    reject = 0

    for x in xrange(k):
        print 'loop:', x+1

        print '*'*50
        print 'Loading the model...'
        model = load_model(current_path)

        print '*'*50
        print 'Predict labels on retrain data...'
        train_pred = binary_predict(model, imgs_train_raw)

        print '*'*50
        print 'Calculate rand error...'
        current_rand = np.mean(rand_error_to_img(imgs_train_label, train_pred))
        print 'current_rand:', current_rand

        if prev_rand < current_rand:
            reject += 1
            print 'prev_rand:', prev_rand, 'current_rand:', current_rand
            print 'Re-traing rejected. reject:', reject
            model = load_model(prev_path)
            train_pred = binary_predict(model, imgs_train_raw)
        else:
            prev_rand = current_rand
            model.save(prev_path)

        print '*'*50
        print 'Part A'
        print '*'*50

        a_train_raw, a_train_label = make_A_array(imgs_train_raw, imgs_train_label, train_pred)

        print '*'*50
        print 'Fitting model...'
        print '*'*50

        ## After each epoch if validation_acc is best, save the model
        checkpoint = ModelCheckpoint(current_path, monitor='val_acc', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')

        ## train
        hist = model.fit(a_train_raw, a_train_label, batch_size = 16, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                validation_data=[imgs_test_raw, imgs_test_label], callbacks=[checkpoint, early_stopping])

        print '*'*50
        print 'Part B'
        print '*'*50

        model = load_model(current_path)
        test_pred = model.predict(imgs_test_raw, batch_size = 16, verbose=1)
        b_train_raw, b_train_label = make_B_array(imgs_test_raw, imgs_test_label, test_pred)

        print '*'*50
        print 'Fitting model...'
        print '*'*50

        ## train
        hist = model.fit(b_train_raw, b_train_label, batch_size = 16, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                validation_data=[imgs_test_raw, imgs_test_label], callbacks=[checkpoint, early_stopping])

        model.save(current_path)

if __name__ == '__main__':
    retrain()
