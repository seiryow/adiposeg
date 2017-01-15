import numpy as np


def flatten_img(y):
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


def flatten_patch(y):
    tmp = np.zeros([y.shape[0]],dtype='uint8')

    l = 0
    for x in xrange(y.shape[0]):
        if y[x][0] == 1:
            tmp[l] = 0
        else:
            tmp[l] = 1
        l += 1

    return tmp


def pixel_error_to_img(y_true, y_pred):
    from sklearn.metrics import hamming_loss

    loss = np.zeros([y_true.shape[0]/16], dtype='float32')

    for x in xrange(y_true.shape[0]/16):
        flatten_true = flatten_img(y_true[x:x+16])
        flatten_pred = flatten_img(y_pred[x:x+16])
        loss[x] = hamming_loss(flatten_true, flatten_pred)

    return loss


def rand_error_to_img(y_true, y_pred):
    from sklearn.metrics.cluster import adjusted_rand_score

    loss = np.zeros([y_true.shape[0]/16], dtype='float32')

    for x in xrange(y_true.shape[0]/16):
        flatten_true = flatten_img(y_true[x:x+16])
        flatten_pred = flatten_img(y_pred[x:x+16])
        loss[x] = 1 - adjusted_rand_score(flatten_true, flatten_pred)

    return loss


def rand_error_to_patch(y_true, y_pred):
    from sklearn.metrics.cluster import adjusted_rand_score

    loss = np.zeros([y_true.shape[0]],dtype='float32')

    for x in xrange(y_true.shape[0]):
        flatten_true = flatten_patch(y_true[x])
        flatten_pred = flatten_patch(y_pred[x])
        loss[x] = 1 - adjusted_rand_score(flatten_true, flatten_pred)

    return loss
