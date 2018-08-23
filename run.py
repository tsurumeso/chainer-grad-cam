import argparse

import chainer
import cv2
import numpy as np

import models
from lib import backprop


p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='images/dog_cat.png')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--arch', '-a',
               choices=['alex', 'vgg', 'resnet'],
               default='vgg')
p.add_argument('--label', '-y', type=int, default=-1)
p.add_argument('--layer', '-l', default='conv5_3')
args = p.parse_args()


if __name__ == '__main__':
    if args.arch == 'alex':
        model = models.Alex()
    elif args.arch == 'vgg':
        model = models.VGG16Layers()
    elif args.arch == 'resnet':
        model = models.ResNet152Layers()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    grad_cam = backprop.GradCAM(model)
    guided_backprop = backprop.GuidedBackprop(model)

    src = cv2.imread(args.input, 1)
    src = cv2.resize(src, (model.size, model.size))
    x = src.astype(np.float32) - np.float32([103.939, 116.779, 123.68])
    x = x.transpose(2, 0, 1)[np.newaxis, :, :, :]

    gcam = grad_cam.generate(x, args.label, args.layer)
    gcam = np.uint8(gcam * 255 / gcam.max())
    gcam = cv2.resize(gcam, (model.size, model.size))
    gbp = guided_backprop.generate(x, args.label, args.layer)

    ggcam = gbp * gcam[:, :, np.newaxis]
    ggcam -= ggcam.min()
    ggcam = 255 * ggcam / ggcam.max()
    cv2.imwrite('ggcam.png', ggcam)

    gbp -= gbp.min()
    gbp = 255 * gbp / gbp.max()
    cv2.imwrite('gbp.png', gbp)

    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = np.float32(src) + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()
    cv2.imwrite('gcam.png', gcam)
