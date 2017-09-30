from __future__ import print_function
import collections
import os

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.dataset import download
from chainer.serializers import npz


class VGG16Layers(chainer.link.Chain):

    def __init__(self):
        super(VGG16Layers, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.fc6 = L.Linear(512 * 7 * 7, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

        _retrieve(
            'VGG_ILSVRC_16_layers.npz',
            'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/'
            'caffe/VGG_ILSVRC_16_layers.caffemodel',
            self)

        self.functions = collections.OrderedDict([
            ('conv1_1', [self.conv1_1, F.relu]),
            ('conv1_2', [self.conv1_2, F.relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, F.relu]),
            ('conv2_2', [self.conv2_2, F.relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, F.relu]),
            ('conv3_2', [self.conv3_2, F.relu]),
            ('conv3_3', [self.conv3_3, F.relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, F.relu]),
            ('conv4_2', [self.conv4_2, F.relu]),
            ('conv4_3', [self.conv4_3, F.relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, F.relu]),
            ('conv5_2', [self.conv5_2, F.relu]),
            ('conv5_3', [self.conv5_3, F.relu]),
            ('pool5', [_max_pooling_2d]),
            ('fc6', [self.fc6, F.relu, F.dropout]),
            ('fc7', [self.fc7, F.relu, F.dropout]),
            ('fc8', [self.fc8]),
            ('prob', [F.softmax]),
        ])

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        npz.save_npz(path_npz, caffemodel, compression=False)

    def __call__(self, x, layers=['prob']):
        h = x
        activations = {'input': x}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def extract(self, image, layers=['fc7']):
        x = prepare(image)
        x = chainer.Variable(self.xp.asarray(x))
        return self(x, layers=layers)


def prepare(image):
    image = image.astype(dtype=np.float32)
    image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    image = image.transpose(2, 0, 1).reshape(1, 3, 224, 224)
    return image


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)


def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    print('Now loading caffemodel (usually it may take few minutes)')
    VGG16Layers.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))
