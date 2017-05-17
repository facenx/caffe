import signal, random, os
import numpy as np
import cv2
from functools import partial
from multiprocessing import Pool
from image_proc import flip


def get_face_cff(img_path, mirror=True, scale=1., img_fnc=None, img_size=None):
    """
    Load face image.
    Its size should correspond to 'blob_shape' parameter passed to Loader initialization.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
    if img_fnc is not None:
        img = img_fnc(img)
    if mirror and random.random() < .5:
        img = flip(img)

    img = img.astype(np.float32)
    img = cv2.resize(img, img_size)
    img -= (104., 117., 123.)
    img *= scale
    img = img.transpose(2, 0, 1)

    return img


class Loader:
    """
    Adapter for loading images.
    """

    def __init__(self, blob_shape, root_folder, source, mirror=True, scale=1., img_fnc=None, processes=0):
        self.root_folder = root_folder
        # load image paths list with their labels
        img_paths = []
        img_labels = []
        # if labels are sparse (test set case) get it from file
        for item in open(source):
            img_path, img_label = item.split(' ')
            img_label = img_label.rstrip()
            img_paths.append(img_path)
            img_labels.append(float(img_label))
        self.img_paths = np.array(img_paths)
        self.img_labels = np.array(img_labels).astype(np.float32)

        # setup service objects
        self.pool = None
        if processes:
            self.pool = Pool(processes=processes)
        self.batch_size = blob_shape[0]
        self.load_fnc = partial(get_face_cff, mirror=mirror, scale=scale, img_fnc=img_fnc, img_size=tuple(blob_shape[2:4]))

        # prepare dst objects
        self.out_images = np.zeros(blob_shape, dtype=np.float32)
        self.out_labels = np.zeros(blob_shape[0], dtype=np.float32)

    def fetch_data(self):

        indexes = np.random.randint(0, high=len(self.img_paths), size=self.batch_size)

        img_paths_batch = [os.path.join(self.root_folder, img_path) for img_path in self.img_paths[indexes]]

        # load images
        if self.pool is not None:
            data = self.pool.map(self.load_fnc, img_paths_batch)
        else:
            data = list(map(self.load_fnc, img_paths_batch))

        for i, img in enumerate(data):
            self.out_images[i][...] = img

        # load labels
        for i, indx in enumerate(indexes):
            self.out_labels[i] = self.img_labels[indx]

        return self.out_images.copy(), self.out_labels.copy()
