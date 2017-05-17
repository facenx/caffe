import numpy as np


# --->>> flip
def flip(img):
    img_out = img[:, ::-1, :].copy()
    return img_out


# --->>> reye
def reye_mode1(img):
    img_out = img[:int(0.5*img.shape[0]), :int(0.5*img.shape[1])].copy()
    return img_out


def reye_mode2(img, lndms):
    c = 0.24  # for 96x96 output size while 200x200 input
    center_ = [(lndms[37 - 1][0] + lndms[40 - 1][0]) / 2, ((lndms[40 - 1][1]) + (lndms[37 - 1][1])) / 2]
    img_size_ = np.array(img.shape[:2]).min()
    img_out = img[max(0, center_[1] - int(c * img_size_)):center_[1] + int(c * img_size_),
                  max(0, center_[0] - int(c * img_size_)):center_[0] + int(c * img_size_)].copy()
    if not img_out.data:
        print('(!) image transformation error')
        img_out = reye_mode1(img, lndms)
    return img_out


def reye_mode3(img, lndms):
    c = 1.5
    size_ = lndms[40 - 1][0] - lndms[37 - 1][0]
    center_ = [lndms[37 - 1][0] + size_ / 2, ((lndms[40 - 1][1]) + (lndms[37 - 1][1])) / 2]
    img_out = img[max(0, center_[1] - int(c * size_)):center_[1] + int(c * size_),
              max(0, center_[0] - int(c * size_)):center_[0] + int(c * size_)].copy()
    if not img_out.data:
        print('(!) image transformation error')
        img_out = reye_mode1(img, lndms)
    return img_out


# --->>> nose
def nose_mode1(img):
    img_out = img[int(0.25 * img.shape[0]):int(0.75 * img.shape[0]), int(0.25 * img.shape[1]):int(0.75 * img.shape[1])].copy()
    return img_out


def nose_mode2(img, lndms):
    c = 0.24  # for 96x96 output size while 200x200 input
    center_ = [(lndms[32 - 1][0] + lndms[36 - 1][0]) / 2, ((lndms[32 - 1][1]) + (lndms[36 - 1][1])) / 2]
    img_size_ = np.array(img.shape[:2]).min()
    img_out = img[max(0, center_[1] - int(c * img_size_)):center_[1] + int(c * img_size_),
              max(0, center_[0] - int(c * img_size_)):center_[0] + int(c * img_size_)].copy()
    if not img_out.data:
        print('(!) image transformation error')
        img_out = nose_mode1(img)
    return img_out


def nose_mode3(img, lndms):
    c = 1.5
    size_ = lndms[36 - 1][0] - lndms[32 - 1][0]
    center_ = [lndms[32 - 1][0] + size_ / 2, ((lndms[36 - 1][1]) + (lndms[32 - 1][1])) / 2]
    img_out = img[max(0, center_[1] - int(c * size_)):center_[1] + int(c * size_),
              max(0, center_[0] - int(c * size_)):center_[0] + int(c * size_)].copy()
    if not img_out.data:
        print('(!) image transformation error')
        img_out = nose_mode1(img)
    return img_out


# --->>> mouth
def mouth_mode1(img):
    img_out = img[int(0.5 * img.shape[0]):, int(0.25 * img.shape[1]):int(0.75 * img.shape[1])].copy()
    return img_out


def mouth_mode2(img, lndms):
    c = 0.24  # for 96x96 output size while 200x200 input
    center_ = [(lndms[49 - 1][0] + lndms[55 - 1][0]) / 2, ((lndms[49 - 1][1]) + (lndms[55 - 1][1])) / 2]
    img_size_ = np.array(img.shape[:2]).min()
    img_out = img[max(0, center_[1] - int(c * img_size_)):center_[1] + int(c * img_size_),
                  max(0, center_[0] - int(c * img_size_)):center_[0] + int(c * img_size_)].copy()
    if not img_out.data:
        print('(!) image transformation error')
        img_out = mouth_mode1(img)
    return img_out


def mouth_mode3(img, lndms):
    c = 0.8
    size_ = lndms[55 - 1][0] - lndms[49 - 1][0]
    center_ = [lndms[49 - 1][0] + size_ / 2, ((lndms[55 - 1][1]) + (lndms[49 - 1][1])) / 2]
    img_out = img[max(0, center_[1] - int(c * size_)):center_[1] + int(c * size_),
              max(0, center_[0] - int(c * size_)):center_[0] + int(c * size_)].copy()
    if not img_out.data:
        print('(!) image transformation error')
        img_out = mouth_mode1(img)
    return img_out


# --->>> eyes
def eyes_mode1(img):
    img_out = img[:int(0.50 * img.shape[0]), :].copy()
    return img_out
