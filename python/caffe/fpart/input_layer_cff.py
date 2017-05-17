import sys
import caffe
import image_proc
from data_loader import Loader


class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        params = eval(self.param_str)

        top[0].reshape(*params['shape'])
        top[1].reshape(params['shape'][0])

        img_fnc = getattr(image_proc, params['img_fnc'])
        self.loader = Loader(params['shape'], params['root_folder'], params['source'],
                             mirror=params['mirror'], scale=params['scale'], img_fnc=img_fnc, processes=params['processes'])

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        imgs, labels = self.loader.fetch_data()
        top[0].data[...] = imgs
        top[1].data[...] = labels

    def backward(self, top, propagate_down, bottom):
        pass


