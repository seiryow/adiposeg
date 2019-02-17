import numpy as np
from PIL import Image
from pprint import pprint


def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.
    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def img_to_array(img, dim_ordering='tf'):
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering: ', dim_ordering)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def array_to_img(x, dim_ordering='tf', scale=True):
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def divide_img(img):
    ## http://d.hatena.ne.jp/Megumi221/20110509/1304953533
    w = img.size[0]
    h = img.size[1]
    rsize = min(w, h)
    box = (0, 0, rsize, rsize)
    region= img.crop(box)

    subregion = list()
    ds = float(rsize)/4.0
    for i in range(4):
        wmin = int(ds*i)
        wmax = int(ds*(i+1))
        for j in range(4):
            k = 4*i + j
            hmin = int(ds*j)
            hmax = int(ds*(j+1))
            box = (wmin, hmin, wmax, hmax)
            subregion.append(region.crop(box))

    return subregion


def combine_img(img_list):
    w = img_list[0].size[0]
    h = img_list[0].size[1]

    combined_img = Image.new("L", (w*4, h*4))

    for x in range(4):
        for y in range(4):
            combined_img.paste(img_list[4*x+y],(w*x,h*y))

    return combined_img


def gamma_img(img_list, img_type):
    tmp_list = list()
    size = len(img_list)

    time = 4
    delta = 2.5 / time

    if img_type == 'raw':
        for x in range(size):
            tmp_list.append(img_list[x])

            gamma = 0.1
            tmp = img_to_array(img_list[x]) ** gamma / 255
            tmp_list.append(array_to_img(tmp, scale=True))

            for t in range(time):
                gamma = delta * (t + 1)
                tmp = img_to_array(img_list[x]) ** gamma / 255
                tmp_list.append(array_to_img(tmp, scale=True))

    if img_type == 'label':
        for x in range(size):
            for y in range(time + 2):
                tmp_list.append(img_list[x])

    return tmp_list


def rotate_img(img_list):
    tmp_list = list()
    size = len(img_list)

    for x in range(size):
        tmp_list.append(img_list[x])

    for x in range(size):
        tmp = img_list[x].transpose(Image.ROTATE_90)
        tmp_list.append(tmp)

    for x in range(size):
        tmp = img_list[x].transpose(Image.ROTATE_180)
        tmp_list.append(tmp)

    for x in range(size):
        tmp = img_list[x].transpose(Image.ROTATE_270)
        tmp_list.append(tmp)

    return tmp_list


def flip_img(img_list):
    tmp_list = list()
    size = len(img_list)

    for x in range(size):
        tmp_list.append(img_list[x])

    for x in range(size):
        flip = img_list[x].transpose(Image.FLIP_LEFT_RIGHT)
        tmp_list.append(flip)

    return tmp_list
