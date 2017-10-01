import argparse
import copy

import chainer
import cv2
import numpy as np

from lib import backprop
import models


def main():
    model = models.VGG16Layers()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    grad_cam = backprop.GradCAM(model, args.label)
    guided_backprop = backprop.GuidedBackprop(copy.deepcopy(model), args.label)

    img = cv2.imread(args.input, 1)
    img = cv2.resize(img, (224, 224))
    gcam = grad_cam.generate(img, args.layer)
    gbp = guided_backprop.generate(img, args.layer)

    ggcam = gbp * gcam[:, :, np.newaxis]
    ggcam -= ggcam.min()
    ggcam = 255 * ggcam / ggcam.max()
    cv2.imwrite('ggcam.png', ggcam)

    gbp -= gbp.min()
    gbp = 255 * gbp / gbp.max()
    cv2.imwrite('gbp.png', gbp)

    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = np.float32(img) + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()
    cv2.imwrite('gcam.png', gcam)


p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='images/dog_cat.png')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--label', '-l', type=int, default=-1)
p.add_argument('--layer', default='conv5_3')
args = p.parse_args()


if __name__ == '__main__':
    main()
