回复不及时请联系这个邮箱哈：zhihao_bj@139.com

# G2R-ShadowNet
[From Shadow Generation to Shadow Removal.](https://arxiv.org/abs/2103.12997)

## Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.1+ & tochvision
- scikit-image

## Training code and data

Generate the training data using the original ISTD dataset and the code ```gettraindata.m```

or download the training data from: GoogleDrive: [ISTD](https://drive.google.com/file/d/1az6HF6VTD6w1AcEFzW4RstAEZgjTqzPB/view?usp=sharing) | BaiduNetdisk: [ISTD](https://pan.baidu.com/s/1TppO4RqehJBh-TEIseYErA) (Access code: 1111)

Your `~/PAISTD8/` folder should look like this
```
PAISTD8
├── train/
    ├── train_A/
    │   └── 90-1.png
    │   └── ...
    ├── train_B/
    │   └── ...
    └── ...
```

## Testing masks produced by BDRAR

GoogleDrive: [ISTD](https://drive.google.com/file/d/1fx7PODULpfRD6dsatvpsNpKoHKeYks7J/view?usp=sharing)

BaiduNetdisk: [ISTD](https://pan.baidu.com/s/1iPh-oR_gttrIkm72S0H-ug) (Access code: 1111)


## Train and test on the adjusted ISTD dataset
Train 
1. Set the path of the dataset in ```train.py```
2. Run ```train.py```

Test   
1. Set the paths of the dataset and saved models ```(netG_1.pth)``` and ```(netG_2.pth)``` in ```test.py```
2. Run ```test.py```

## Evaluate
1. Set the paths of the shadow removal results and the dataset in ```evaluate.m```
2. Run ```evaluate.m```

## The Best Models on ISTD

GoogleDrive: [ISTD](https://drive.google.com/file/d/1uSqGRbSXm12dpNIfaSsVYdQW4ifYbgw0/view?usp=sharing)

BaiduNetdisk: [ISTD](https://pan.baidu.com/s/1QJx-ccmE4-pQWK0v9nA00g) (Access code: 1111)

 
## Results of G2R-ShadowNet on ISTD

GoogleDrive: [ISTD](https://drive.google.com/file/d/1qDhKWeihp6dqzINrtdkwc4SIkzx42yx3/view?usp=sharing)

BaiduNetdisk: [ISTD](https://pan.baidu.com/s/1fQ4f6zFBkqUwnimA4k1M1A) (Access code: 1111)

## Results of G2R-ShadowNet-_Sup._ on ISTD

GoogleDrive: [ISTD](https://drive.google.com/file/d/1Q1dIyGIOxPpi9yk8ibrcIPIH0G2IqE2e/view?usp=sharing)

BaiduNetdisk: [ISTD](https://pan.baidu.com/s/1JWvz0KVjkbQmcoHENuGc4w) (Access code: 1111)


## ISTD Results (size: 480x640)
| Method | Shadow Region | Non-shadow Region | All |
|:-----|:-----:|:-----:|------|
| [Le & Samaras (ECCV20)](https://github.com/lmhieu612/FSS2SR) | 11.3 | 3.7 | 4.8 |
| G2R-ShadowNet (Ours) | 9.6 | 3.8 | 4.7 |

Results in shadow and non-shadow regions are computed on each image first and then compute the average of all images in terms of RMSE.

## Acknowledgments
Code is implemented based on [Mask-ShadowGAN](https://github.com/xw-hu/Mask-ShadowGAN) and [LG-ShadowNet](https://github.com/hhqweasd/LG-ShadowNet).

All codes will be released to public soon.

```
@inproceedings{liu2021from,
  title={From Shadow Generation to Shadow Removal},
  author={Liu, Zhihao and Yin, Hui and Wu, Xinyi and Wu, Zhenyao and Mi, Yang and Wang, Song},
  booktitle={CVPR},
  year={2021}
}
```


