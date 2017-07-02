import os
import numpy as np
from ios import check_file_list
import sys

input_path = 'input/'

img_rows = 512
img_cols = 512

data_augument = True

## in this function binary label will be converted for keras
def categorize_label(imgs, data_augument=False):
    clabels = np.zeros([imgs.shape[0], 1*imgs.shape[2]*imgs.shape[3], 2], dtype='uint8')

    i = 0
    for img_array in imgs:
        if data_augument == True and int(i/16) % 12 !=0:
            # this mention is used for speeding up,
            # but may cause some bugs.
            # if you change a data augumentation method
            # and this program does not work,
            # first of all you doubt here.
            index = i - 16 * (int(i/16) % 12)
            clabels[i] = clabels[index]
        else:
            l = 0
            for j in xrange(imgs.shape[2]):
                for k in xrange(imgs.shape[3]):
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

## load imgs and return img list
def get_img_list(file_path, file_list, img_type, data_augument = False):
    from image import load_img

    tmp_list = list()
    for file in file_list:
        img = load_img(os.path.join(file_path, file), grayscale=True, target_size=(img_rows, img_cols))
        tmp_list.append(img)

        if data_augument == True:
            augmentated_list = data_augumentation(img, img_type=img_type)

            for tmp in augmentated_list:
                tmp_list.append(tmp)

    return tmp_list

## divide each img in tmp_list and return them in the form of array
def get_divided_img_array(imgs, tmp_list):
    from image import img_to_array, divide_img

    i=0
    for tmp in tmp_list:
        divided_tmp = divide_img(tmp)

        for patch in divided_tmp:
            imgs[i] = img_to_array(patch) / 255
            i += 1

    return imgs


def data_augumentation(img, img_type):
    from image import gamma_img, rotate_img, flip_img

    augmentated_list = list()
    augmentated_list.append(img)

    augmentated_list = gamma_img(augmentated_list, img_type)
    augmentated_list = rotate_img(augmentated_list)
    augmentated_list = flip_img(augmentated_list)

    del augmentated_list[0]

    return augmentated_list


def make_test_and_val_list(tmp_raw_list, tmp_label_list, tmp_name_list):
    import random

    total = len(tmp_raw_list)

    index = random.sample(xrange(total), int(total/2))
    sort = index
    sort.sort()

    test_raw_list = list()
    test_label_list = list()
    test_name_list = list()
    val_raw_list = list()
    val_label_list = list()

    j = 0
    for i in xrange(total):
        if i == sort[j]:
            val_raw_list.append(tmp_raw_list[i])
            val_label_list.append(tmp_label_list[i])
            if(j < int(total/2)-1):
                j += 1
        else:
            test_raw_list.append(tmp_raw_list[i])
            test_label_list.append(tmp_label_list[i])
            test_name_list.append(tmp_name_list[i])

    return test_raw_list, test_label_list, test_name_list, val_raw_list, val_label_list


def make_train_array(train_path, data_augument = False):
    from ios import get_file_list

    print '*'*30
    print 'make train_raw array...'
    print '*'*30
    file_path = os.path.join(train_path,'raw/')
    file_list, name_list = get_file_list(file_path)

    tmp_raw_list = get_img_list(file_path, file_list, img_type='raw', data_augument=data_augument)

    total = len(tmp_raw_list)
    imgs_raw = np.zeros([total*16, 1, img_rows/4, img_cols/4], dtype='float32')

    imgs_raw = get_divided_img_array(imgs_raw, tmp_raw_list)

    print '*'*30
    print 'make train_label array...'
    print '*'*30
    file_path = os.path.join(train_path,'label/')
    file_list, name_list = get_file_list(file_path)

    tmp_label_list = get_img_list(file_path, file_list, img_type='label', data_augument=data_augument)

    total = len(tmp_label_list)
    imgs_label = np.zeros([total*16, 1, img_rows/4, img_cols/4], dtype='float32')

    imgs_label = get_divided_img_array(imgs_label, tmp_label_list)

    imgs_label = categorize_label(imgs_label, data_augument=data_augument)

    return imgs_raw, imgs_label


def make_test_array(test_path):
    from ios import get_file_list

    print '*'*30
    print 'make test_raw array...'
    print '*'*30

    file_path = os.path.join(test_path,'raw/')
    file_list, name_list = get_file_list(file_path)

    tmp_raw_list = get_img_list(file_path, file_list, img_type='raw', data_augument=False)

    print '*'*30
    print 'make test_label array...'
    print '*'*30

    file_path = os.path.join(test_path,'label/')
    file_list, name_list = get_file_list(file_path)

    tmp_label_list = get_img_list(file_path, file_list, img_type='label', data_augument=False)

    test_raw_list, test_label_list, test_name_list, val_raw_list, val_label_list = make_test_and_val_list(tmp_raw_list, tmp_label_list, name_list)

    total_test = len(test_raw_list)*16
    test_raw = np.zeros([total_test, 1, img_rows/4, img_cols/4], dtype='float32')
    test_label = np.zeros([total_test, 1, img_rows/4, img_cols/4], dtype='float32')

    total_val = len(val_raw_list)*16
    val_raw = np.zeros([total_val, 1, img_rows/4, img_cols/4], dtype='float32')
    val_label = np.zeros([total_val, 1, img_rows/4, img_cols/4], dtype='float32')

    test_raw = get_divided_img_array(test_raw, test_raw_list)
    test_label = get_divided_img_array(test_label, test_label_list)
    val_raw = get_divided_img_array(val_raw, val_raw_list)
    val_label = get_divided_img_array(val_label, val_label_list)

    test_label = categorize_label(test_label, data_augument=False)
    val_label = categorize_label(val_label, data_augument=False)

    return test_raw, test_label, test_name_list, val_raw, val_label


## this function is for debug
def visualize_patches(data_path, output_path, img_type):
    from image import combine_img, array_to_img
    from predict import clabels_to_img

    if img_type not in {'raw', 'label'}:
        raise ValueError('Invalid img_type:', img_type)

    data = np.load(data_path)

    if img_type == 'label':
        imgs = clabels_to_img(data, 512, 512)
    else:
        imgs = data

    total = len(imgs)

    tmp_list = list()
    combined_imgs = np.zeros([total/16, 1, imgs.shape[2], imgs.shape[3]])

    for img_array in imgs:
        img_array *= 255
        img = array_to_img(img_array, scale=False)
        tmp_list.append(img)

    for x in xrange(total/16):
        combined_img = combine_img(tmp_list[16*x:16*(x+1)])

        data_path = os.path.join(output_path,'img'+str(x)+'.png')

        combined_img.save(data_path)
        print 'save image:',x+1,'/',total/16


if __name__ == '__main__':
    print '*'*50
    print 'Load training images...'
    print '*'*50
    traindir = sys.argv[1]+'/'
    train_path = os.path.join(traindir, 'train')
    if check_file_list(train_path)==False:
        raise ValueError('Labels do not match with raws.')

    imgs_train_raw, imgs_train_label = make_train_array(train_path, data_augument)

    imgs_retrain_raw, imgs_retrain_label = make_train_array(train_path, data_augument=False)

    print '*'*50
    print 'Load test images...'
    print '*'*50
    test_path = './test/'
    if check_file_list(test_path)==False:
        raise ValueError('Labels do not match with raws.')

    imgs_test_raw, imgs_test_label, imgs_test_name, val_test_raw, val_test_label = make_test_array(test_path)

    print '*'*50

    print 'Save loaded images to numpy files...'

    np.save(traindir + 'train_raw.npy', imgs_train_raw)
    np.save(traindir + 'train_label.npy', imgs_train_label)
    np.save(traindir + 'retrain_raw.npy', imgs_retrain_raw)
    np.save(traindir + 'retrain_label.npy', imgs_retrain_label)
    np.save(traindir + 'test_raw.npy', imgs_test_raw)
    np.save(traindir + 'test_label.npy', imgs_test_label)
    np.save(traindir + 'test_name.npy', imgs_test_name)
    np.save(traindir + 'val_test_raw.npy', val_test_raw)
    np.save(traindir + 'val_test_label.npy', val_test_label)

    print 'imgs_train_raw:', imgs_train_raw.shape
    print 'imgs_train_label:', imgs_train_label.shape
    print 'imgs_retrain_raw:', imgs_retrain_raw.shape
    print 'imgs_retrain_label:', imgs_retrain_label.shape
    print 'imgs_test_raw:', imgs_test_raw.shape
    print 'imgs_test_label:', imgs_test_label.shape
    print 'val_test_raw:', val_test_raw.shape
    print 'val_test_label:', val_test_label.shape

    print '*'*50

    print 'Done.'

    ## for test display
    #visualize_patches('train_raw.npy', 'test/train/raw/' , img_type='raw')
    #visualize_patches('train_label.npy', 'test/train/label/', img_type='label')
    #visualize_patches('test_raw.npy', 'test/test/raw/', img_type='raw')
    #visualize_patches('test_label.npy', 'test/test/label/', img_type='label')
    #visualize_patches('val_test_raw.npy', 'test/val/raw/', img_type='raw')
    #visualize_patches('val_test_label.npy', 'test/val/label/', img_type='label')
