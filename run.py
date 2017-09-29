import argparse

import chainer
import chainer.functions as F
from chainer.links import VGG16Layers

import cv2
import numpy as np


def main():
    xp = np
    model = VGG16Layers()
    if args.gpu >= 0:
        xp = chainer.cuda.cupy
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = cv2.imread(args.input, 1)
    acts = model.extract((img[:, :, ::-1],), layers=[args.layer, 'prob'])

    one_hot = xp.zeros((1, 1000), dtype=np.float32)
    if args.label == -1:
        one_hot[:, acts['prob'].data.argmax()] = 1
    else:
        one_hot[:, args.label] = 1
    loss = F.sum(chainer.Variable(one_hot) * acts['prob'])
    loss.backward(retain_grad=True)

    weights = xp.mean(acts[args.layer].grad, axis=(2, 3))
    cam = xp.tensordot(weights[0], acts[args.layer].data[0], axes=(0, 0))
    cam = (cam > 0) * cam / cam.max()
    cam = chainer.cuda.to_cpu(cam * 255)
    cam = cv2.resize(np.uint8(cam), (224, 224))

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img = np.float32(img) + np.float32(heatmap)
    img = 255 * img / img.max()
    cv2.imwrite('heatmap.png', img)


p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='images/boxer_cat.png')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--label', '-l', type=int, default=-1)
p.add_argument('--layer', default='conv5_3')
args = p.parse_args()


if __name__ == '__main__':
    main()
