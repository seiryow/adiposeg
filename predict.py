import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import array_to_img
from datetime import datetime

output_path = 'output/'
model_path = 'weights/unet.hdf5'
#model_path = 'weights/2016-12-27/17-59-42/weights.383.hdf5'

img_rows = 128
img_cols = 128

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

    ## for binary output
    output = np.round(output)

    return output


def clabels_to_img(clabels):
    total = clabels.shape[0]
    labels = np.ndarray((total, 1, img_rows, img_cols))

    for i in range(0,total):
        j = 0
        k = 0
        for l in range(0,1*img_rows*img_cols):
            if clabels[i][l][0] == 0:
                labels[i][0][j][k] = 1
            else:
                labels[i][0][j][k] = 0
            k += 1
            if k == img_cols:
                k = 0
                j += 1

    return labels


def visualize(imgs):
    imgs_test_name = np.load('test_name.npy')

    total = len(imgs)

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
    for img_array in imgs:
        np.set_printoptions(threshold=np.inf)

        img = array_to_img(img_array)
        img.save(os.path.join(dir_path, imgs_test_name[i]))
        i+=1
        print 'save label image:',i,'/',total

    print 'Done.'


if __name__ == '__main__':
    output = predict()
    output_reshaped = clabels_to_img(output)
    visualize(output_reshaped)
