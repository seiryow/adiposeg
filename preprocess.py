import os
import numpy as np
from image import load_img, array_to_img, img_to_array

input_path = 'input/'

img_rows = 512
img_cols = 512


def get_file_list(path):
    import imghdr
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


## this function is now unused
def check_label_list(path, file_list):
    for file in file_list:
        target = os.path.join(path,file).replace("\\", "/")
        if not os.path.exists(target):
            print 'Not exist the label of ',file
            return False

    return True


def categorize_label(imgs, data_augument=False):
    clabels = np.zeros([imgs.shape[0], 1*imgs.shape[2]*imgs.shape[3], 2], dtype='uint8')

    i = 0
    for img_array in imgs:
        if data_augument == True and int(i/16) % 12 !=0:
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


def data_augumentation(img, img_type):
    from image import gamma_img, rotate_img, flip_img

    augmentated_list = list()
    augmentated_list.append(img)

    augmentated_list = gamma_img(augmentated_list, img_type)
    augmentated_list = rotate_img(augmentated_list)
    augmentated_list = flip_img(augmentated_list)

    del augmentated_list[0]

    return augmentated_list


def make_image_array(path, img_type, data_augument=False):
    from image import divide_img

    if img_type not in {'raw', 'label'}:
        raise ValueError('Invalid img_type:', img_type)

    if img_type == 'raw':
        print 'Load raw images ...'
        file_path = os.path.join(path,'raw/')
        file_list, name_list = get_file_list(file_path)

    if img_type == 'label':
        print 'Load label images ...'
        file_path = os.path.join(path,'label/')
        file_list, name_list = get_file_list(file_path)

    tmp_list = list()

    for file in file_list:
        img = load_img(os.path.join(file_path, file), grayscale=True, target_size=(img_rows,img_cols))
        tmp_list.append(img)

        if data_augument == True:
            augmentated_list = data_augumentation(img, img_type=img_type)

            for tmp in augmentated_list:
                tmp_list.append(tmp)

    total = len(tmp_list) * 16

    if img_type == 'raw':
        imgs = np.zeros([total, 1, img_rows/4, img_cols/4], dtype='float32')

    if img_type == 'label':
        imgs = np.zeros([total, 1, img_rows/4, img_cols/4], dtype='uint8')

    i=0
    for tmp in tmp_list:
        divided_tmp = divide_img(tmp)

        for patch in divided_tmp:
            imgs[i] = img_to_array(patch) / 255
            i += 1

    if img_type == 'label':
        imgs = categorize_label(imgs, data_augument=data_augument)

    print 'loaded', len(file_list), 'images.'

    return imgs, name_list

## this function is for debug
def visualize(data_path, img_type):
    from image import combine_img
    from predict2 import clabels_to_img

    if img_type not in {'raw', 'label'}:
        raise ValueError('Invalid img_type:', img_type)

    test_path = 'test/'
    data = np.load(data_path)

    if img_type == 'label':
        imgs = clabels_to_img(data)
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

        data_path = os.path.join(test_path,'img'+str(x)+'.png')

        combined_img.save(data_path)
        print 'save image:',x+1,'/',total/16


if __name__ == '__main__':
    print '*'*30
    print 'Load training images...'
    print '*'*30
    train_path = os.path.join(input_path,'train/')
    imgs_train_raw, tmp_name = make_image_array(train_path, img_type='raw', data_augument=True)
    imgs_train_label, tmp_name = make_image_array(train_path, img_type='label', data_augument=True)

    print '*'*30
    print 'Load test images...'
    print '*'*30
    test_path = os.path.join(input_path,'test/')
    imgs_test_raw, tmp_name = make_image_array(test_path, img_type='raw', data_augument=False)
    imgs_test_label, tmp_name = make_image_array(test_path, img_type='label', data_augument=False)

    print '*'*30

    print 'Save loaded images to numpy files...'

    np.save('train_raw.npy', imgs_train_raw)
    np.save('train_label.npy', imgs_train_label)
    np.save('test_raw.npy', imgs_test_raw)
    np.save('test_label.npy', imgs_test_label)
    np.save('test_name.npy', tmp_name)

    print 'imgs_train_raw:',imgs_train_raw.shape
    print 'imgs_train_label:',imgs_train_label.shape
    print 'imgs_test_raw:',imgs_test_raw.shape
    print 'imgs_test_label:',imgs_test_label.shape

    print '*'*30

    print 'Done.'

    ## for test display
    #visualize('train_raw.npy', img_type='raw')
    visualize('train_label.npy', img_type='label')
    #visualize('test_raw.npy',img_type='raw')
    #visualize('test_label.npy', img_type='label')
