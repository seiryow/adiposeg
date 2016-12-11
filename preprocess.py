import os
import imghdr
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras import backend as K

K.set_image_dim_ordering('th')  # th(Theano) or tf(Tensorflow)

input_path = 'input/'

img_rows = 512
img_cols = 512

def get_raw_list(path):
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

def make_image_array(path):
    raw_path = os.path.join(path,'raw/')
    label_path = os.path.join(path,'label/')

    print 'Load raw images ...'
    image_list,name_list = get_raw_list(raw_path)

    total = len(image_list)
    img_head = load_img(os.path.join(raw_path,image_list[0]),grayscale=True,target_size=(img_rows,img_cols))
    img_head = img_to_array(img_head)
    imgs = np.ndarray((total, img_head.shape[0],img_head.shape[1],img_head.shape[2]))
    imgs = imgs.astype('float32')

    i = 0
    for image in image_list:
        img = load_img(os.path.join(raw_path,image),grayscale=True,target_size=(img_rows,img_cols))
        img_array = img_to_array(img)
        img_array /= 255

        imgs[i] = img_array
        i+=1
        print 'load raw image:',i,'/',total

    print 'Load label images ...'
    if check_label_list(label_path,image_list) == False:
        return imgs, None, name_list
    else:
        labels = np.ndarray((total, img_head.shape[0],img_head.shape[1],img_head.shape[2]))
        labels = labels.astype('float32')

        i = 0
        for image in image_list:
            label = load_img(os.path.join(label_path,image),grayscale=True,target_size=(img_rows,img_cols))
            label_array = img_to_array(label)
            label_array /= 255

            labels[i] = label_array
            i+=1
            print 'load label image:',i,'/',total

    return imgs, labels, name_list

if __name__ == '__main__':
    print '*'*30
    print 'Load training images...'
    print '*'*30
    imgs_train_raw, imgs_train_label, imgs_train_name = make_image_array(os.path.join(input_path,'train/'))

    np.save('train_raw.npy', imgs_train_raw)
    np.save('train_label.npy', imgs_train_label)

    print imgs_train_raw.shape

    print '*'*30
    print 'Load test images...'
    print '*'*30
    imgs_test_raw, imgs_test_label, imgs_test_name = make_image_array(os.path.join(input_path,'test/'))
    np.save('test_raw.npy', imgs_test_raw)
    np.save('test_label.npy', imgs_test_label)
    np.save('test_name.npy',imgs_test_name)

    print 'Done.'
