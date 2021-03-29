# G2R-ShadowNet
[From Shadow Generation to Shadow Removal.](https://arxiv.org/abs/2103.12997)

```
@toappear{liu2021from,
  title={From Shadow Generation to Shadow Removal},
  author={Liu, Zhihao and Yin, Hui and Wu, Xinyi and Wu, Zhenyao and Mi, Yang and Wang, Song},
  journal=CVPR,
  year={2021}
}
```

## Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.1+ & tochvision
- scikit-image

## Results of G2R-ShadowNet

GoogleDrive: [ISTD](https://drive.google.com/file/d/1qDhKWeihp6dqzINrtdkwc4SIkzx42yx3/view?usp=sharing)

BaiduNetdisk: [ISTD](https://pan.baidu.com/s/1fQ4f6zFBkqUwnimA4k1M1A) (Access code: 1111)


# ISTD Results (size: 480x640)
| Method | Shadow Region | Non-shadow Region | All |
|:-----|:-----:|:-----:|------|
| [Le & Samaras (ECCV20)](https://github.com/hhqweasd/LG-ShadowNet) | 11.3 | 3.7 | 4.8 |
| G2R-ShadowNet (Ours) | 9.6 | 3.8 | 4.7 |

Results in shadow and non-shadow regions are computed on each image first and then compute the average of all images in terms of RMSE.

## Acknowledgments
Code is implemented based on [Mask-ShadowGAN](https://github.com/xw-hu/Mask-ShadowGAN) and [LG-ShadowNet](https://github.com/hhqweasd/LG-ShadowNet).

All codes will be released to public soon.

## 其他说明
有问题可以联系我
+86 18410949118
刘志浩
