# chainer-grad-cam

## Summery

![](https://github.com/tsurumeso/chainer-grad-cam/blob/master/images/summery.png?raw=true)

## Requirements

- Chainer
- Cupy (Optional, CUDA library required)
- Opencv
- Pillow

## Usage
```
python run.py --input images/boxer_cat.png --label 242 --gpu 0
python run.py --input images/boxer_cat.png --label 282 --gpu 0
```

## References

- [1] Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, Dhruv Batra,
"Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization",
https://arxiv.org/abs/1610.02391
