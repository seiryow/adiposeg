import os
import imghdr
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from predict import clabels_to_img
from predict import visualize as vs

input_path = 'input/'

img_rows = 128
img_cols = 128

def get_file_list(path):
    file_list = []
    name_list = []
    for (root, dirs, files) in os.walk(path):
        for file in files:
            target = os.path.join(root,file).replace("\\", "/")
            if os.path.isfile(target):
                if imghdr.what(target) != None :
                    target = target.replace(path,'')
                    file_list.append(target)
                    name_list.append(file)

    return file_list, name_list


def check_label_list(path,file_list):
    for file in file_list:
        target = os.path.join(path,file).replace("\\", "/")
        if not os.path.exists(target):
            print 'Not exist the label of ',file
            return False

    return True


def categorize_label(clabels,label,i):
    label_array = img_to_array(label) / 255
    label_array = label_array.astype('uint8')

    l = 0
    for j in range(0, img_rows):
        for k in range(0, img_cols):
            if label_array[0][j][k] == 0:
                clabels[i][l][0] = 1
                clabels[i][l][1] = 0
            else :
                clabels[i][l][0] = 0
                clabels[i][l][1] = 1
            l += 1

    return clabels


def gamma_image_train(imgs, img, i):
    gamma = 0.1
    imgs[i] = img_to_array(img) ** gamma / 255
    i += 1

    gamma = 0.25
    while gamma <= 2.5:
        tmp_array = img_to_array(img) ** gamma / 255
        imgs[i] = tmp_array
        i += 1
        gamma += 0.25

    return imgs, i


def gamma_image_test(clabels, label, i):
    for k in range(0,11):
        if k == 0:
            clabels = categorize_label(clabels, label, i)
        else:
            clabels[i+k] = clabels[i]

    i += 11
    return clabels, i


def rotate_image_train(imgs, img, i):
    ## rotate 90 and get its gamma
    tmp = img.transpose(Image.ROTATE_90)
    imgs[i] = img_to_array(tmp) / 255
    i += 1

    imgs, i = gamma_image_train(imgs, tmp, i)

    ## rotate 180 and get its gamma
    tmp = img.transpose(Image.ROTATE_180)
    imgs[i] = img_to_array(tmp) / 255
    i += 1

    imgs, i = gamma_image_train(imgs, tmp, i)

    ## rotate 270 and get its gamma
    tmp = img.transpose(Image.ROTATE_270)
    imgs[i] = img_to_array(tmp) / 255
    i += 1

    ## gamma
    imgs, i = gamma_image_train(imgs, tmp, i)

    return imgs, i


def rotate_image_test(clabels, label, i):
    from PIL import Image

    # rotate 90 and get its gamma
    tmp = label.transpose(Image.ROTATE_90)
    clabels = categorize_label(clabels, tmp, i)
    i += 1

    clabels, i = gamma_image_test(clabels, tmp, i)

    # rotate 180 and get its gamma
    tmp = label.transpose(Image.ROTATE_180)
    clabels = categorize_label(clabels, tmp, i)
    i += 1

    clabels, i = gamma_image_test(clabels, tmp, i)

    # rotate 270 and get its gamma
    tmp = label.transpose(Image.ROTATE_270)
    clabels = categorize_label(clabels, tmp, i)
    i += 1

    clabels, i = gamma_image_test(clabels, tmp, i)

    return clabels, i


def data_augumentation_train(imgs, img, i):
    from PIL import Image

    ## original
    # get gamma
    imgs, i = gamma_image_train(imgs, img, i)

    # rotate and get its gamma
    imgs, i = rotate_image_train(imgs, img, i)

    ## flip
    # make flip
    flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    imgs[i] = img_to_array(flip) / 255
    i+= 1

    # get gamma
    imgs, i = gamma_image_train(imgs, flip, i)

    # rotate and get its gamma
    imgs, i = rotate_image_train(imgs, flip, i)

    return imgs, i


def data_augumentation_test(clabels, label, i):
    from PIL import Image

    ## original
    # get gamma
    clabels, i = gamma_image_test(clabels, label, i)

    # rotate and get its gamma
    clabels, i = rotate_image_test(clabels, label, i)

    ## flip
    # make flip
    flip = label.transpose(Image.FLIP_LEFT_RIGHT)
    clabels = categorize_label(clabels, flip, i)
    i += 1

    # get gamma
    clabels, i = gamma_image_test(clabels, flip, i)

    # rotate and get its gamma
    clabels, i = rotate_image_test(clabels, flip, i)

    return clabels, i


def make_image_array(path,data_augumentation_flag):
    raw_path = os.path.join(path,'raw/')
    label_path = os.path.join(path,'label/')

    print 'Load raw images ...'
    image_list,name_list = get_file_list(raw_path)

    total = len(image_list)

    if data_augumentation_flag:
        total = total * 2 * 4 * 12

    imgs = np.ndarray((total, 1, img_rows, img_cols))
    imgs = imgs.astype('float32')

    i = 0
    for image in image_list:
        img = load_img(os.path.join(raw_path,image), grayscale=True, target_size=(img_rows,img_cols))
        img_array = img_to_array(img) / 255

        imgs[i] = img_array
        i+=1

        if data_augumentation_flag:
            imgs, i = data_augumentation_train(imgs, img, i)

        print 'load raw image:',i,'/',total

    print 'Load label images ...'
    if check_label_list(label_path,image_list) == False:
        return imgs, None, name_list
    else:
        categorized_labels = np.ndarray((total, 1*img_rows*img_cols, 2))
        categorized_labels = categorized_labels.astype('int')

        i = 0
        for image in image_list:
            label = load_img(os.path.join(label_path,image), grayscale=True, target_size=(img_rows,img_cols))
            categorized_labels = categorize_label(categorized_labels, label, i)
            i += 1

            if data_augumentation_flag:
                categorized_labels, i = data_augumentation_test(categorized_labels, label, i)

            print 'load label image:',i,'/',total

    return imgs, categorized_labels, name_list


def visualize(data_path):
    from PIL import Image
    from keras.preprocessing.image import array_to_img
    test_path = 'test'

    imgs = np.load(data_path)
    total = len(imgs)

    i = 0
    for img_array in imgs:
        img = array_to_img(img_array,scale=True)
        data_path = os.path.join(test_path,'img'+str(i)+'.png')
        img.save(data_path)
        i+=1
        if i % 100 == 0:
            print 'save label image:',i,'/',total

    print 'Done.'


if __name__ == '__main__':
    print '*'*30
    print 'Load training images...'
    print '*'*30
    imgs_train_raw, imgs_train_label, imgs_train_name = make_image_array(os.path.join(input_path,'train/'), 1)

    np.save('train_raw.npy', imgs_train_raw)
    np.save('train_label.npy', imgs_train_label)

    print '*'*30
    print 'Load test images...'
    print '*'*30
    imgs_test_raw, imgs_test_label, imgs_test_name = make_image_array(os.path.join(input_path,'test/'), 0)
    np.save('test_raw.npy', imgs_test_raw)
    np.save('test_label.npy', imgs_test_label)
    np.save('test_name.npy',imgs_test_name)

    print '*'*30

    print 'imgs_train_raw:',imgs_train_raw.shape
    print 'imgs_train_label:',imgs_train_label.shape
    print 'imgs_test_raw:',imgs_test_raw.shape
    print 'imgs_test_label:',imgs_test_label.shape

    print '*'*30

    print 'Done.'

    #vs(clabels_to_img(imgs_test_label))

    #visualize('train_raw.npy')
    #visualize('train_label.npy')
    #visualize('test_raw.npy')
    #visualize('test_label.npy')
