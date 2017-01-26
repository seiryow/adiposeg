import os
import numpy as np
from ios import make_output_dir

output_path = 'output/'
#model_path = 'weights/unet.hdf5'
#model_path = 'weights/2017-01-25/22-37-24/weights.005.hdf5'
model_path = 'weights/2017-01-26/16-01-52/current.hdf5'

def binary_predict(model, y):
    output = model.predict(y, batch_size=16, verbose=1)

    output = np.round(output)
    output = output.astype('uint8')

    return output


def clabels_to_img(clabels, img_rows, img_cols):
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


def make_error_file(dir_path, imgs_test_name, y_true, y_pred):
    import csv
    from metrics import pixel_error_to_img, rand_error_to_img

    f = open(os.path.join(dir_path,'error.csv'), 'ab')
    csvWriter = csv.writer(f)

    pix_err = pixel_error_to_img(y_true, y_pred)
    rand_err = rand_error_to_img(y_true, y_pred)

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

    tmp = list()
    tmp.append('mean')
    tmp.append(np.mean(pix_err))
    tmp.append(np.mean(rand_err))
    csvWriter.writerow(tmp)

    f.close()

    print 'Wrote calculated error to the file.'


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
            print 'save label image:', x+1, '/', total

        combined_imgs[x] = img_to_array(combined_img) / 255

    return combined_imgs


def visualize_ts(dir_path, imgs_test_name, imgs_true, imgs_pred):
    from image import array_to_img
    ts_path = 'ts/'
    dir_path = os.path.join(dir_path, ts_path)
    os.mkdir(dir_path)

    print 'Calculate threat score...'

    for x in xrange(imgs_test_name.shape[0]):
        ts_img = np.zeros([3, imgs_true.shape[2], imgs_true.shape[3]], dtype='uint8')

        ts_img[0] = imgs_true[x][0]
        ts_img[1] = imgs_pred[x][0]
        ts_img[2] = imgs_pred[x][0]

        ts_img *= 255
        ts = array_to_img(ts_img,scale=False)
        ts.save(os.path.join(dir_path,imgs_test_name[x]))


if __name__ == '__main__':
    from keras.models import load_model

    print '*'*50
    print 'Loading and preprocessing test data...'
    print '*'*50
    imgs_test_raw = np.load('test_raw.npy')
    imgs_test_label = np.load('test_label.npy')
    imgs_test_name = np.load('test_name.npy')

    print '*'*50
    print 'Loading built model and trained weights...'
    print '*'*50
    model = load_model(model_path)

    print '*'*50
    print 'Predict labels on test data...'
    print '*'*50
    pred_clabels = binary_predict(model, imgs_test_raw)

    print '*'*50
    print 'Calculate pixel error and rand error...'
    print '*'*50
    dir_path = make_output_dir(output_path)

    make_error_file(dir_path, imgs_test_name, imgs_test_label, pred_clabels)

    print '*'*50
    print 'Decategorize labels...'
    print '*'*50
    img_rows = imgs_test_raw.shape[2]*4
    img_cols = imgs_test_raw.shape[3]*4

    true_labels = clabels_to_img(imgs_test_label, img_rows, img_cols)
    pred_labels = clabels_to_img(pred_clabels, img_rows, img_cols)

    print '*'*50
    print 'Output images...'
    print '*'*50

    imgs_pred = visualize(dir_path, imgs_test_name, pred_labels, visualize=True)
    imgs_true = visualize(dir_path, imgs_test_name, true_labels, visualize=False)

    visualize_ts(dir_path, imgs_test_name, imgs_true, imgs_pred)

    print 'Done.'
