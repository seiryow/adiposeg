import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ios import make_output_dir
from metrics import rand_error_to_img, rand_error_to_patch, pixel_error_to_img
from predict import binary_predict, clabels_to_img

weight_path = 'weights/'
model_path = 'weights/unet.hdf5'
#model_path = 'weights/2017-02-23/05-05-54/weights.005.hdf5'
output_path = 'output/'

nb_epoch = 10
loop = 7

def clabel_to_img(clabel, img_rows, img_cols):
    img = np.zeros((1, img_rows, img_cols), dtype = 'uint8')

    j = 0
    k = 0
    for l in range(clabel.shape[0]):
        if clabel[l][0] == 0:
            img[0][j][k] = 1
        else:
            img[0][j][k] = 0
        k += 1
        if k == img_cols:
            k = 0
            j += 1

    return img


def categorize_label(imgs, data_augument=False):
    clabels = np.zeros([imgs.shape[0], 1*imgs.shape[2]*imgs.shape[3], 2], dtype='uint8')

    i = 0
    for img_array in imgs:
        l = 0
        for j in range(imgs.shape[2]):
            for k in range(imgs.shape[3]):
                if img_array[0][j][k] == 0:
                    clabels[i][l][0] = 1
                    clabels[i][l][1] = 0
                else:
                    clabels[i][l][0] = 0
                    clabels[i][l][1] = 1
                l += 1
        i += 1

        if i%1000 == 0 or i == clabels.shape[0]:
            print 'Categorized', i, '/',  clabels.shape[0]

    return clabels


def rotate_img(img_list):
    tmp_list = list()
    size = len(img_list)

    for x in range(size):
        tmp_list.append(img_list[x])

        tmp = img_list[x].transpose(Image.ROTATE_90)
        tmp_list.append(tmp)
        tmp = img_list[x].transpose(Image.ROTATE_180)
        tmp_list.append(tmp)
        tmp = img_list[x].transpose(Image.ROTATE_270)
        tmp_list.append(tmp)

    return tmp_list


def flip_img(img_list):
    tmp_list = list()
    size = len(img_list)

    for x in range(size):
        tmp_list.append(img_list[x])

        flip = img_list[x].transpose(Image.FLIP_LEFT_RIGHT)
        tmp_list.append(flip)

    return tmp_list


def get_random_index(sort):
    import random

    r = random.random()

    bottom = sort[0]
    if r < bottom:
        return 0

    for x in range(sort.shape[0]-1):
        top = sort[x+1]
        if bottom <= r and r < top:
            return x+1
        bottom = top


def data_augumentation(imgs, img_type):
    from image import array_to_img, img_to_array
    import cv2

    tmp = list()

    for x in range(imgs.shape[0]):
        img = imgs[x]
        if img_type == 'clabel':
            img = clabel_to_img(img, 128, 128)
        tmp.append(array_to_img(img))

    augmentated_list = rotate_img(tmp)
    augmentated_list = flip_img(augmentated_list)

    total = len(augmentated_list)
    augmentated_imgs = np.zeros([total, 1, 128, 128], dtype='float32')

    for x in range(total):
        if img_type == 'clabel':
            data_path = os.path.join('test/retrain/label/img'+str(x)+'.png')
            augmentated_list[x].save(data_path)
        else:
            data_path = os.path.join('test/retrain/raw/img'+str(x)+'.png')
            augmentated_list[x].save(data_path)

        augmentated_imgs[x] = img_to_array(augmentated_list[x]) / 255

    if img_type == 'raw':
        return augmentated_imgs
    if img_type == 'clabel':
        augmentated_clabels = categorize_label(augmentated_imgs)
        return augmentated_clabels


def make_retrain_array(imgs_test_raw, imgs_test_label, test_pred):
    import math

    test_total = test_pred.shape[0]
    size = test_pred.shape[1]
    prob = np.zeros([test_total], dtype="float32")
    for x in range(test_total):
        tmp = 0
        for i in range(size):
            tmp += math.fabs(test_pred[x][i][1] - 0.5)
        prob[x] = tmp/size

    #sort = np.sort(prob)
    #sort -= sort[0]
    #sort /= sort[sort.shape[0]-1]

    index = np.argsort(prob)

    ntotal = index.shape[0] / 100 * 25
    if ntotal % 16 != 0:
        ntotal += 16 - (ntotal % 16)              #for deconv

    new_train_raw = np.zeros([ntotal, 1, imgs_test_raw.shape[2], imgs_test_raw.shape[3]], dtype='float32')
    new_train_label = np.zeros([ntotal, test_pred.shape[1], 2], dtype='uint8')

    for x in range(ntotal):
        #i = get_random_index(sort)
        #new_train_raw[x] = imgs_test_raw[index[i]]
        #new_train_label[x] = np.round(test_pred[index[i]])

        #data_path = os.path.join('test/retrain/raw/img'+str(x)+'.png')
        #array_to_img(imgs_test_raw[index[i]]).save(data_path)
        #data_path = os.path.join('test/retrain/label/img'+str(x)+'.png')
        #array_to_img(imgs_test_raw[index[i]]).save(data_path)

        new_train_raw[x] = imgs_test_raw[index[ntotal-1-x]]
        new_train_label[x] = np.round(test_pred[index[ntotal-1-x]])

    new_train_raw = data_augumentation(new_train_raw, 'raw')
    new_train_label = data_augumentation(new_train_label, 'clabel')

    return new_train_raw, new_train_label


def retrain():
    import shutil

    print '*'*50
    print 'Loading train data...'
    print '*'*50
    imgs_retrain_raw = np.load('retrain_raw.npy')
    imgs_retrain_label = np.load('retrain_label.npy')

    imgs_test_raw = np.load('test_raw.npy')
    imgs_test_label = np.load('test_label.npy')

    val_test_raw = np.load('val_test_raw.npy')
    val_test_label = np.load('val_test_label.npy')

    dir_path = make_output_dir(weight_path)

    prev_path = os.path.join(dir_path, 'prev.hdf5')
    current_path = os.path.join(dir_path, 'unet.hdf5')

    shutil.copyfile(model_path, current_path)

    prev_rand = 1
    reject = 0

    checkpoint = ModelCheckpoint(current_path, monitor='val_acc', save_best_only=True)

    for x in range(loop):
        print '*'*50
        print 'Loop:', x
        print '*'*50

        print '*'*50
        print 'Loading the model...'
        model = load_model(current_path)

        print '*'*50
        print 'Predict labels on retrain data...'
        val_pred = binary_predict(model, val_test_raw)

        print '*'*50
        print 'Calculate rand error...'
        current_pix = np.mean(pixel_error_to_img(val_test_label, val_pred))
        current_rand = np.mean(rand_error_to_img(val_test_label, val_pred))
        print '(current_pix, current_rand):', (current_pix, current_rand)

        test_pred = binary_predict(model, imgs_test_raw)
        test_pix = np.mean(pixel_error_to_img(imgs_test_label, test_pred))
        test_rand = np.mean(rand_error_to_img(imgs_test_label, test_pred))
        print '(test_pix, test_rand):', (test_pix, test_rand)

        # adopt judge
        if prev_rand < current_rand:
            reject += 1
            print 'prev_rand:', prev_rand, 'current_rand:', current_rand
            print 'Re-traing rejected. reject:', reject
            model = load_model(prev_path)
        else:
            prev_rand = current_rand
            model.save(prev_path)

        test_pred = model.predict(imgs_test_raw, batch_size = 16, verbose=1)
        new_train_raw, new_train_label = make_retrain_array(imgs_test_raw, imgs_test_label, test_pred)

        print '*'*50
        print 'Fitting model...'
        print '*'*50

        new_train_raw = np.concatenate((imgs_retrain_raw, new_train_raw), axis=0)
        new_train_label = np.concatenate((imgs_retrain_label, new_train_label), axis=0)

        ## train
        hist = model.fit(new_train_raw, new_train_label, batch_size = 16, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                validation_data=[val_test_raw, val_test_label], callbacks=[checkpoint])

        ## adopt judge
        if x == loop-1:
            print '*'*50
            print 'Predict labels on retrain data...'
            val_pred = binary_predict(model, val_test_raw)

            print '*'*50
            print 'Calculate rand error...'
            current_pix = np.mean(pixel_error_to_img(val_test_label, val_pred))
            current_rand = np.mean(rand_error_to_img(val_test_label, val_pred))
            print '(current_pix, current_rand):', (current_pix, current_rand)

            if prev_rand < current_rand:
                reject += 1
                print 'prev_rand:', prev_rand, 'current_rand:', current_rand
                print 'Re-traing rejected. reject:', reject
                shutil.copyfile(prev_path, current_path)

            ## for test
            model = load_model(current_path)
            test_pred = binary_predict(model, imgs_test_raw)
            test_pix = np.mean(pixel_error_to_img(imgs_test_label, test_pred))
            test_rand = np.mean(rand_error_to_img(imgs_test_label, test_pred))
            print '(test_pix, test_rand):', (test_pix, test_rand)

    shutil.copyfile(current_path, os.path.join(weight_path,'unet.hdf5'))
    os.remove(prev_path)

if __name__ == '__main__':
    retrain()
