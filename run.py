import argparse

import chainer
import chainer.functions as F
from chainer.links import VGG16Layers

import cv2
import numpy as np


def main():
    model = VGG16Layers()
    img = cv2.imread(args.input, 1)
    acts = model.extract((img[:, :, ::-1],), layers=[args.layer, 'prob'])

    one_hot = np.zeros((1, 1000), dtype=np.float32)
    if args.label == -1:
        one_hot[:, acts['prob'].data.argmax()] = 1
    else:
        one_hot[:, args.label] = 1
    loss = F.sum(chainer.Variable(one_hot) * acts['prob'])
    loss.backward(retain_grad=True)

    weights = np.mean(acts[args.layer].grad.data, axis=(2, 3))
    cam = np.zeros(acts[args.layer].data.shape[2:])
    for w, fmap in zip(weights[0], acts[args.layer].data[0]):
        cam += w * fmap
    cam /= cam.max()
    cam = (cam > 0) * cam
    cam = cv2.resize(np.uint8(cam * 255), (224, 224))

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img = img * 0.5 + heatmap + 0.5
    cv2.imwrite('heatmap.png', img)


p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='images/boxer_cat.png')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--label', '-l', type=int, default=-1)
p.add_argument('--layer', default='conv5_3')
args = p.parse_args()


if __name__ == '__main__':
    main()
