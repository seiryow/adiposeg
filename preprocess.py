import os
import numpy as np
from ios import check_file_list, get_file_list
import sys
import argparse
from distutils.util import strtobool
from pprint import pprint

img_rows = 512
img_cols = 512

data_augment = True
data_augment = False

## in this function binary label will be converted for keras
def categorize_label(imgs, data_augment=False):
    return np.stack((imgs == 0, imgs != 0), axis=-1).reshape((imgs.shape[0], -1, 2))

## load imgs and return img list
def get_img_list(file_path, file_list, img_type, data_augment=False):
    from image import load_img

    tmp_list = list()
    for filename in file_list:
        img = load_img(os.path.join(file_path, filename), grayscale=True,
                                    target_size=(img_rows, img_cols))
        tmp_list.append(img)

        if data_augment:
            tmp_list += data_augmentation(img, img_type=img_type)

    return tmp_list

## divide each img in tmp_list and return them in the form of array
def get_divided_img_array(tmp_list):
    from image import img_to_array, divide_img

    imgs = [img_to_array(patch) / 255
                for tmp in tmp_list
                for patch in divide_img(tmp)]

    return np.stack(imgs, axis=0)


def data_augmentation(img, img_type):
    from image import gamma_img, rotate_img, flip_img

    augmentated_list = list()
    augmentated_list.append(img)

    augmentated_list = gamma_img(augmentated_list, img_type)
    augmentated_list = rotate_img(augmentated_list)
    augmentated_list = flip_img(augmentated_list)

    del augmentated_list[0]

    return augmentated_list


def make_array(data_path, data_augment=False):
    print('*'*50)
    print('Load images from %s...' % data_path)
    print('*'*50)
    pprint (data_path)
    if not check_file_list(data_path):
        raise ValueError('%s Labels do not match with raws.' % data_path)

    print('*'*30)
    print('%s make raw array...' % data_path)
    print('*'*30)
    file_path = os.path.join(data_path,'raw/')
    file_list, name_list = get_file_list(file_path)

    tmp_raw_list = get_img_list(file_path, file_list, img_type='raw', data_augment=data_augment)

    total = len(tmp_raw_list)
    imgs_raw = np.zeros([total*16, int(img_rows/4), int(img_cols/4), 1], dtype='float32')
    pprint(tmp_raw_list)
    imgs_raw = get_divided_img_array(tmp_raw_list)

    print('*'*30)
    print('%s make label array...' % data_path)
    print('*'*30)
    file_path = os.path.join(data_path,'label/')
    file_list, name_list = get_file_list(file_path)

    tmp_label_list = get_img_list(file_path, file_list, img_type='label', data_augment=data_augment)

    total = len(tmp_label_list)
    imgs_label = np.zeros([total*16, int(img_rows/4), int(img_cols/4), 1], dtype='float32')

    imgs_label = get_divided_img_array(tmp_label_list)

    imgs_label = categorize_label(imgs_label, data_augment=data_augment)

    return imgs_raw, imgs_label, name_list


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

    for x in range(total/16):
        combined_img = combine_img(tmp_list[16*x:16*(x+1)])

        data_path = os.path.join(output_path,'img'+str(x)+'.png')

        combined_img.save(data_path)
        print('save image:',x+1,'/',total/16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess test and train images')
    parser.add_argument('dataset_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('data_augment', type=strtobool, default=True, help='Use data augment')
    args = parser.parse_args()
    data_augment = args.data_augment
    print (data_augment)
    train_path = os.path.join(args.dataset_dir, 'train')
    val_path = os.path.join(args.dataset_dir, 'val')
    test_path = os.path.join(args.dataset_dir, 'test')

    imgs_train_raw, imgs_train_label, imgs_train_name = make_array(train_path, data_augment)
    imgs_retrain_raw, imgs_retrain_label, _ = make_array(train_path, data_augment=False)
    imgs_test_raw, imgs_test_label, imgs_test_name = make_array(test_path, False)
    imgs_val_raw, imgs_val_label, _ = make_array(val_path, False)

    print('*'*50)

    print('Save loaded images to numpy files...')

    np.save(os.path.join(args.dataset_dir, 'train_raw.npy'), imgs_train_raw)
    np.save(os.path.join(args.dataset_dir, 'train_label.npy'), imgs_train_label)
    np.save(os.path.join(args.dataset_dir, 'train_name.npy'), imgs_train_name)
    np.save(os.path.join(args.dataset_dir, 'retrain_raw.npy'), imgs_retrain_raw)
    np.save(os.path.join(args.dataset_dir, 'retrain_label.npy'), imgs_retrain_label)
    np.save(os.path.join(args.dataset_dir, 'test_raw.npy'), imgs_test_raw)
    np.save(os.path.join(args.dataset_dir, 'test_label.npy'), imgs_test_label)
    np.save(os.path.join(args.dataset_dir, 'test_name.npy'), imgs_test_name)
    np.save(os.path.join(args.dataset_dir, 'val_test_raw.npy'), imgs_val_raw)
    np.save(os.path.join(args.dataset_dir, 'val_test_label.npy'), imgs_val_label)

    print('imgs_train_raw:', imgs_train_raw.shape)
    print('imgs_train_label:', imgs_train_label.shape)
    print('imgs_retrain_raw:', imgs_retrain_raw.shape)
    print('imgs_retrain_label:', imgs_retrain_label.shape)
    print('imgs_test_raw:', imgs_test_raw.shape)
    print('imgs_test_label:', imgs_test_label.shape)
    print('imgs_val_raw:', imgs_val_raw.shape)
    print('imgs_val_label:', imgs_val_label.shape)

    print('*'*50)

    print('Done.')

    ## for test display
    #visualize_patches('train_raw.npy', 'test/train/raw/' , img_type='raw')
    #visualize_patches('train_label.npy', 'test/train/label/', img_type='label')
    #visualize_patches('test_raw.npy', 'test/test/raw/', img_type='raw')
    #visualize_patches('test_label.npy', 'test/test/label/', img_type='label')
    #visualize_patches('val_test_raw.npy', 'test/val/raw/', img_type='raw')
    #visualize_patches('val_test_label.npy', 'test/val/label/', img_type='label')
