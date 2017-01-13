import os
import numpy as np
from keras.models import load_model
from datetime import datetime

output_path = 'output/'
#model_path = 'weights/unet.hdf5'
model_path = 'weights/2017-01-10/07-16-17/weights.007.hdf5'

img_rows = 512
img_cols = 512

def predict(model, imgs_test_raw):
    output = model.predict(imgs_test_raw, batch_size=1, verbose=1)

    output = np.round(output)
    output = output.astype('uint8')

    return output


def clabels_to_img(clabels):
    total = clabels.shape[0]
    imgs = np.zeros((total, 1, img_rows/4, img_cols/4), dtype = 'uint8')

    for i in xrange(total):
        j = 0
        k = 0
        for l in xrange(clabels.shape[1]):
            if clabels[i][l][0] == 0:
                imgs[i][0][j][k] = 1
            else:
                imgs[i][0][j][k] = 0
            k += 1
            if k == img_cols/4:
                k = 0
                j += 1

        if (i+1) % 1000 == 0 or i+1 == total:
            print 'Decategorized', i+1, '/', total

    return imgs


def visualize(dir_path, imgs_test_name, imgs, visualize=True):
    from image import combine_img, array_to_img, img_to_array

    tmp_list = list()

    for img_array in imgs:
        img_array *= 255
        img = array_to_img(img_array, scale=False)
        tmp_list.append(img)

    total = imgs_test_name.shape[0]
    combined_imgs = np.zeros([total, 1, imgs.shape[2]*4, imgs.shape[3]*4], dtype = 'uint8')

    for x in xrange(total):
        combined_img = combine_img(tmp_list[16*x:16*(x+1)])
        if visualize == True:
            combined_img.save(os.path.join(dir_path, imgs_test_name[x]))
            print 'save label image:',x+1,'/',total

        combined_imgs[x] = img_to_array(combined_img) / 255

    return combined_imgs


def flatten(y):
    tmp = np.zeros([y.shape[0]*y.shape[1]],dtype='uint8')

    l = 0
    for x in xrange(y.shape[0]):
        for k in xrange(y.shape[1]):
            if y[x][k][0] == 1:
                tmp[l] = 0
            else:
                tmp[l] = 1
            l += 1

    return tmp


def calculate_pixel_error(y_true, y_pred):
    from sklearn.metrics import hamming_loss

    loss = np.zeros([y_true.shape[0]/16],dtype='float32')

    for x in xrange(y_true.shape[0]/16):
        flatten_true = flatten(y_true[x:x+16])
        flatten_pred = flatten(y_pred[x:x+16])
        loss[x] = hamming_loss(flatten_true, flatten_pred)

    return loss


def calculate_rand_error(y_true, y_pred):
    from sklearn.metrics.cluster import adjusted_rand_score

    loss = np.zeros([y_true.shape[0]/16],dtype='float32')

    for x in xrange(y_true.shape[0]/16):
        flatten_true = flatten(y_true[x:x+16])
        flatten_pred = flatten(y_pred[x:x+16])
        loss[x] = 1 - adjusted_rand_score(flatten_true, flatten_pred)

    return loss


def make_error_file(dir_path, imgs_test_name, y_true, y_pred):
    import csv

    f = open(os.path.join(dir_path,'error.csv'), 'ab')
    csvWriter = csv.writer(f)

    pix_err = calculate_pixel_error(y_true, y_pred)
    rand_err = calculate_rand_error(y_true, y_pred)

    hist = np.zeros([imgs_test_name.shape[0],2],dtype='float32')
    history = list()
    s = "img,pixel_err,rand_err"
    l = s.split(",")
    csvWriter.writerow(l)

    for x in xrange(hist.shape[0]):
        tmp = list()
        tmp.append(imgs_test_name[x])
        tmp.append(pix_err[x])
        tmp.append(rand_err[x])
        csvWriter.writerow(tmp)

    print 'Made Error File.'


def visualize_ts(dir_path, imgs_test_name, imgs_true, imgs_pred):
    from image import array_to_img
    ts_path = 'ts/'
    dir_path = os.path.join(dir_path, ts_path)
    os.mkdir(dir_path)

    print 'Calculate Threat Score...'

    for x in xrange(imgs_test_name.shape[0]):
        ts_img = np.zeros([3, imgs_true.shape[2], imgs_true.shape[3]], dtype='uint8')

        ts_img[0] = imgs_true[x][0]
        ts_img[1] = imgs_pred[x][0]
        ts_img[2] = imgs_pred[x][0]

        ts_img *= 255
        ts = array_to_img(ts_img,scale=False)
        ts.save(os.path.join(dir_path,imgs_test_name[x]))

    print 'Done.'


if __name__ == '__main__':

    print '*'*30
    print 'Loading and preprocessing test data...'
    print '*'*30
    imgs_test_raw = np.load('test_raw.npy')
    imgs_test_label = np.load('test_label.npy')
    imgs_test_name = np.load('test_name.npy')

    print '*'*30
    print 'Loading built model and trained weights...'
    print '*'*30
    model = load_model(model_path)

    print '*'*30
    print 'Predict labels on test data...'
    print '*'*30
    pred_clabels = predict(model, imgs_test_raw)

    time = datetime.now()

    day_dir = time.strftime('%Y-%m-%d')
    dir_path = os.path.join(output_path,day_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    time_dir = time.strftime('%H-%M-%S')
    dir_path = os.path.join(dir_path,time_dir)
    os.mkdir(dir_path)

    print '*'*30
    print 'Calculate pixel error and rand error...'
    print '*'*30
    make_error_file(dir_path, imgs_test_name, imgs_test_label, pred_clabels)

    print '*'*30
    print 'Decategorize labels...'
    print '*'*30
    true_labels = clabels_to_img(imgs_test_label)
    pred_labels = clabels_to_img(pred_clabels)

    print '*'*30
    print 'Output images...'
    print '*'*30

    imgs_pred = visualize(dir_path, imgs_test_name, pred_labels, visualize=True)
    imgs_true = visualize(dir_path, imgs_test_name, true_labels, visualize=False)

    visualize_ts(dir_path, imgs_test_name, imgs_true, imgs_pred)
