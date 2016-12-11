import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import array_to_img
from datetime import datetime
from keras import backend as K

K.set_image_dim_ordering('th')  # th(Theano) or tf(Tensorflow)

output_path = 'output/'
model_path = 'weights/unet.hdf5'

def predict():
    print '*'*30
    print 'Loading and preprocessing test data...'
    print '*'*30
    imgs_test_raw = np.load('test_raw.npy')
    imgs_test_label = np.load('test_label.npy')

    print '*'*30
    print 'Loading built model and trained weights...'
    print '*'*30
    model = load_model(model_path)

    print '*'*30
    print 'Predicting labels on test data...'
    print '*'*30
    output = model.predict(imgs_test_raw, batch_size=1, verbose=1)

    return output

def visualize(img_array):
    imgs_test_name = np.load('test_name.npy')
    test_len = len(imgs_test_name)

    total = len(img_array)

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
        if(test_len==total):
            img.save(os.path.join(dir_path,imgs_test_name[i]))
        else:
            img.save(os.path.join(dir_path,'img',i,'.png'))
        i+=1
        print 'save label image:',i,'/',total

    print 'Done.'

if __name__ == '__main__':
    output = predict()
    visualize(output)
