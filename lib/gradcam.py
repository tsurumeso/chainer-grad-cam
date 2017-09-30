import cv2
import numpy as np

import chainer
import chainer.functions as F

from chainer import cuda
from chainer import utils


class GuidedReLU(chainer.function.Function):

    def forward(self, x):
        xp = chainer.cuda.get_array_module(x[0])
        self.retain_inputs(())
        self.retain_outputs((0,))
        y = xp.maximum(x[0], 0)

        return y,

    def backward_cpu(self, x, gy):
        y = self.output_data[0]
        return utils.force_array(gy[0] * (y > 0) * (gy[0] > 0)),

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = (y > 0 && gy > 0) ? gy : (T)0',
            'relu_bwd')(y, gy[0])
        return gx,


class BaseBackProp(object):

    def __init__(self, model, label):
        self.model = model
        self.label = label
        self.xp = model.xp

    def backward(self, x, layer):
        with chainer.using_config('train', False):
            acts = self.model.extract(x, layers=[layer, 'prob'])

        one_hot = self.xp.zeros((1, 1000), dtype=np.float32)
        if self.label == -1:
            one_hot[:, acts['prob'].data.argmax()] = 1
        else:
            one_hot[:, self.label] = 1

        self.model.cleargrads()
        loss = F.sum(chainer.Variable(one_hot) * acts['prob'])
        loss.backward(retain_grad=True)

        return acts


class GradCAM(BaseBackProp):

    def generate(self, x, layer):
        acts = self.backward(x, layer)
        weights = self.xp.mean(acts[layer].grad, axis=(2, 3))
        gcam = self.xp.tensordot(weights[0], acts[layer].data[0], axes=(0, 0))
        gcam = (gcam > 0) * gcam / gcam.max()
        gcam = chainer.cuda.to_cpu(gcam * 255)
        gcam = cv2.resize(np.uint8(gcam), (224, 224))

        return gcam


class GuidedBackprop(BaseBackProp):

    def __init__(self, model, label):
        super(GuidedBackprop, self).__init__(model, label)
        for key, funcs in model.functions.items():
            for i in range(len(funcs)):
                if funcs[i] is F.relu:
                    funcs[i] = GuidedReLU()

    def generate(self, x, layer):
        acts = self.backward(x, layer)
        gbp = chainer.cuda.to_cpu(acts['input'].grad[0])
        gbp = gbp.transpose(1, 2, 0)

        return gbp
