import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import array_to_img
from datetime import datetime
from keras import backend as K

from train import preprocess

K.set_image_dim_ordering('th')  # th(Theano) or tf(Tensorflow)

output_path = 'output/'

def load_test_data():
    test_raw = np.load('test_raw.npy')
    test_label = np.load('test_label.npy')

    return test_raw, test_label

def load_test_name():
    test_name = np.load('test_name.npy')

    return test_name

def predict():
    print '*'*30
    print 'Loading and preprocessing test data...'
    print '*'*30
    imgs_test_raw, imgs_test_label = load_test_data()
    imgs_test_raw = preprocess(imgs_test_raw)

    print '*'*30
    print 'Loading built model and trained weights...'
    print '*'*30
    model = load_model('unet.h5')

    print '*'*30
    print 'Predicting labels on test data...'
    print '*'*30
    output = model.predict(imgs_test_raw, verbose=1)

    return output

def visualize(output):
    imgs_test_name = load_test_name()

    total = len(output)

    time = datetime.now()

    day_dir = time.strftime('%Y-%m-%d')
    dir_path = os.path.join(output_path,day_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    time_dir = time.strftime('%H-%M-%S')
    dir_path = os.path.join(dir_path,time_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    i = 0
    for img_array in output:
        img = array_to_img(img_array,scale=True)
        img.save(os.path.join(dir_path,imgs_test_name[i]))
        i+=1
        print 'save label image:',i,'/',total

    print 'Done.'

if __name__ == '__main__':
    output = predict()
    visualize(output)
