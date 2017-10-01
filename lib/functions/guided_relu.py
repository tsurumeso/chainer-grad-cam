import chainer
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
