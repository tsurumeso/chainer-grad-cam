import sys

import chainer
import chainer.functions as F
from chainer.links import VGG16Layers

import cv2
import numpy as np


model = VGG16Layers()
img = cv2.imread(sys.argv[1], 1)
acts = model.extract((img[:, :, ::-1],), layers=['conv5_3', 'prob'])

one_hot = np.zeros((1, 1000), dtype=np.float32)
one_hot[:, 282] = 1
one_hot = F.sum(chainer.Variable(one_hot) * acts['prob'])
one_hot.backward(retain_grad=True)

weights = np.mean(acts['conv5_3'].grad.data, axis=(2, 3))
cam = np.zeros(acts['conv5_3'].data.shape[2:])
for w, fmap in zip(weights[0], acts['conv5_3'].data[0]):
    cam += w * fmap
cam /= cam.max()
cam = (cam > 0) * cam
cam = cv2.resize(np.uint8(cam * 255), (224, 224))

heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
img = img * 0.5 + heatmap + 0.5
cv2.imwrite('heatmap.png', img)
