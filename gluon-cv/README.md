# CSER

### This is part of the python implementation of the paper "CSER: Communication-efficient SGD with Error Reset".

This repository is a modified version of GluonCV
-----------------

### Requirements

The following python packages needs to be installed:

1. MXNet (a modified version in the folder "incubator-mxnet", read the instructions inside)
2. Horovod

**To install MXNet, please follow the instructions in README.md in the "incubator-mxnet" folder

To install this repository, run:
```bash
python3 setup.py install --user
```

### Run the CIFAR-100 demo

* Train WideResNet on CIFAR-100 with 8 workers, CSER, with compression ratios and H: 8,4,2, batch size is 16 for each worker:
```bash
cd scripts/classification/cifar
horovodrun -np 8 -hostfile HOSTFILE python3 train_cifar100_hvd_ersgd2_v2.py \
  --model cifar_wideresnet40_8 --optimizer nag --lr 0.1 --lr-decay 0.2 \
  --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
  --input-sparse-1 8 --output-sparse-1 1 --layer-sparse-1 1 \
  --input-sparse-2 4 --output-sparse-2 1 --layer-sparse-2 1 \
  --local-sgd-interval 2
```

* The demo of all the algorithms could be found in the following scripts in the folder "scripts/classification/cifar":
1. SGD: exp_SGD.sh
2. EF-SGD: exp_EFSGD.sh
3. QSparse-local-SGD: exp_QSparseLocalSGD.sh
4. CSER: exp_CSER.sh
5. CSEA: exp_CSEA.sh
6. CSER-PL: exp_CSERPL.sh
   
* To run the demo scripts, the user must prepare the hostfile in Horovod format (see https://horovod.readthedocs.io/en/stable/running_include.html), and then run:
```bash
bash exp_CSER.sh hostfile
```

### Run the ImageNet demo

* The users need to prepare ImageNet dataset according to the following tutorials:
1. https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html
2. https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html 



* Train ResNet50 on ImageNet with 8 workers, CSER, with compression ratios and H: 8,4,2, batch size is 32 for each worker:
```bash
cd scripts/classification/imagenet
horovodrun -np 8 -hostfile hostfile python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec \
    --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec \
    --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   \
    --lr 0.1 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    --trainer sgd 
```

* The demo of all the algorithms could be found in the following scripts in the folder "scripts/classification/imagenet":
1. SGD: exp_imagenet_SGD.sh
2. EF-SGD: exp_imagenet_EFSGD.sh
3. QSparse-local-SGD: exp_imagenet_QSparseLocalSGD.sh
4. CSER: exp_imagenet_CSER.sh
5. CSEA: exp_imagenet_CSEA.sh
6. CSER-PL: exp_imagenet_CSERPL.sh
   
* To run the demo scripts, the user must prepare the hostfile in Horovod format (see https://horovod.readthedocs.io/en/stable/running_include.html), and then run:
```bash
bash exp_imagenet_CSER.sh hostfile
```

--------------------------------------------------------------------
--------------------------------------------------------------------


# Gluon CV Toolkit

[![Build Status](http://ci.mxnet.io/buildStatus/icon?job=gluon-cv%2Fmaster)](http://ci.mxnet.io/job/gluon-cv/job/master/)
[![GitHub license](docs/_static/apache2.svg)](./LICENSE)
[![Code Coverage](http://gluon-cv.mxnet.io/coverage.svg?)](http://gluon-cv.mxnet.io/coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/gluoncv.svg)](https://pypi.python.org/pypi/gluoncv)
[![PyPI Pre-release](https://img.shields.io/badge/pypi--prerelease-v0.7.0-ff69b4.svg)](https://pypi.org/project/gluoncv/#history)
[![Downloads](http://pepy.tech/badge/gluoncv)](http://pepy.tech/project/gluoncv)

| [Installation](https://gluon-cv.mxnet.io/install.html) | [Documentation](https://gluon-cv.mxnet.io) | [Tutorials](https://gluon-cv.mxnet.io/tutorials/index.html) |

GluonCV provides implementations of the state-of-the-art (SOTA) deep learning models in computer vision.

It is designed for engineers, researchers, and
students to fast prototype products and research ideas based on these
models. This toolkit offers four main features:

1. Training scripts to reproduce SOTA results reported in research papers
2. A large number of pre-trained models
3. Carefully designed APIs that greatly reduce the implementation complexity
4. Community supports

# Demo

<div align="center">
    <img src="docs/_static/short_demo.gif">
</div>

<br>

Check the HD video at [Youtube](https://www.youtube.com/watch?v=nfpouVAzXt0) or [Bilibili](https://www.bilibili.com/video/av55619231).

# Supported Applications

| Application  | Illustration  | Available Models |
|:-----------------------:|:---:|:---:|
| [Image Classification:](https://gluon-cv.mxnet.io/model_zoo/classification.html) <br/>recognize an object in an image.  | <a href="https://gluon-cv.mxnet.io/model_zoo/classification.html"><img  src="docs/_static/image-classification.png" alt="classification" height="200"/></a>  | 50+ models, including <br/><a href="https://gluon-cv.mxnet.io/model_zoo/classification.html#resnet">ResNet</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/classification.html#mobilenet">MobileNet</a>, <br/><a href="https://gluon-cv.mxnet.io/model_zoo/classification.html#densenet">DenseNet</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/classification.html#vgg">VGG</a>, ... |
| [Object Detection:](https://gluon-cv.mxnet.io/model_zoo/detection.html) <br/>detect multiple objects with their <br/> bounding boxes in an image.     |  <a href="https://gluon-cv.mxnet.io/model_zoo/detection.html"><img src="docs/_static/object-detection.png" alt="detection" height="200"/></a> | <a href="https://gluon-cv.mxnet.io/model_zoo/detection.html#faster-rcnn">Faster RCNN</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/detection.html#ssd">SSD</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/detection.html#yolo-v3">Yolo-v3</a> |
| [Semantic Segmentation:](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation) <br/>associate each pixel of an image <br/> with a categorical label. |  <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation"><img src="docs/_static/semantic-segmentation.png" alt="semantic" height="200"/></a> | <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation">FCN</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation">PSP</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation">ICNet</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation">DeepLab-v3</a> |
| [Instance Segmentation:](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#instance-segmentation) <br/>detect objects and associate <br/> each pixel inside object area with an <br/> instance label. | <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#instance-segmentation"><img src="docs/_static/instance-segmentation.png" alt="instance" height="200"/></a> | <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#instance-segmentation">Mask RCNN</a>|
| [Pose Estimation:](https://gluon-cv.mxnet.io/model_zoo/pose.html) <br/>detect human pose <br/> from images. | <a href="https://gluon-cv.mxnet.io/model_zoo/pose.html"><img src="docs/_static/pose-estimation.svg" alt="pose" height="200"/></a> | <a href="https://gluon-cv.mxnet.io/model_zoo/pose.html#simple-pose-with-resnet">Simple Pose</a>|
| [Video Action Recognition:](https://gluon-cv.mxnet.io/model_zoo/action_recognition.html) <br/>recognize human actions <br/> in a video. | <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html"><img src="docs/_static/action-recognition.png" alt="action_recognition" height="200"/></a> | <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">TSN</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">C3D</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">I3D</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">P3D</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">R3D</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">R2+1D</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">Non-local</a>, <a href="https://gluon-cv.mxnet.io/model_zoo/action_recognition.html">SlowFast</a> |
| [GAN:](https://github.com/dmlc/gluon-cv/tree/master/scripts/gan) <br/>generate visually deceptive images | <a href="https://github.com/dmlc/gluon-cv/tree/master/scripts/gan"><img src="https://github.com/dmlc/gluon-cv/raw/master/scripts/gan/wgan/fake_samples_400000.png" alt="lsun" height="200"/></a> | <a href="https://github.com/dmlc/gluon-cv/tree/master/scripts/gan/wgan">WGAN</a>, <a href="https://github.com/dmlc/gluon-cv/tree/master/scripts/gan/cycle_gan">CycleGAN</a> |
| [Person Re-ID:](https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/baseline) <br/>re-identify pedestrians across scenes | <a href="https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/baseline"><img src="https://user-images.githubusercontent.com/3307514/46702937-f4311800-cbd9-11e8-8eeb-c945ec5643fb.png" alt="re-id" height="160"/></a> |<a href="https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/baseline">Market1501 baseline</a> |

# Installation

GluonCV supports Python 2.7/3.5 or later. The easiest way to install is via pip.

## Stable Release
The following commands install the stable version of GluonCV and MXNet:

```bash
pip install gluoncv --upgrade
pip install mxnet-mkl --upgrade
# if cuda 10.1 is installed
pip install mxnet-cu101mkl --upgrade
```

**The latest stable version of GluonCV is 0.6 and depends on mxnet >= 1.4.0**

## Nightly Release

You may get access to latest features and bug fixes with the following commands which install the nightly build of GluonCV and MXNet:

```bash
pip install gluoncv --pre --upgrade
pip install mxnet-mkl --pre --upgrade
# if cuda 10.1 is installed
pip install mxnet-cu101mkl --pre --upgrade
```

There are multiple versions of MXNet pre-built package available. Please refer to [mxnet packages](https://gluon-crash-course.mxnet.io/mxnet_packages.html) if you need more details about MXNet versions.

# Docs 📖
GluonCV documentation is available at [our website](https://gluon-cv.mxnet.io/index.html).

# Examples

All tutorials are available at [our website](https://gluon-cv.mxnet.io/index.html)!

- [Image Classification](http://gluon-cv.mxnet.io/build/examples_classification/index.html)

- [Object Detection](http://gluon-cv.mxnet.io/build/examples_detection/index.html)

- [Semantic Segmentation](http://gluon-cv.mxnet.io/build/examples_segmentation/index.html)

- [Instance Segmentation](http://gluon-cv.mxnet.io/build/examples_instance/index.html)

- [Video Action Recognition](https://gluon-cv.mxnet.io/build/examples_action_recognition/index.html)

- [Generative Adversarial Network](https://github.com/dmlc/gluon-cv/tree/master/scripts/gan)

- [Person Re-identification](https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/)

# Resources

Check out how to use GluonCV for your own research or projects.

- For background knowledge of deep learning or CV, please refer to the open source book [*Dive into Deep Learning*](http://diveintodeeplearning.org/). If you are new to Gluon, please check out [our 60-minute crash course](http://gluon-crash-course.mxnet.io/).
- For getting started quickly, refer to notebook runnable examples at [Examples](https://gluon-cv.mxnet.io/build/examples_classification/index.html).
- For advanced examples, check out our [Scripts](http://gluon-cv.mxnet.io/master/scripts/index.html).
- For experienced users, check out our [API Notes](https://gluon-cv.mxnet.io/api/data.datasets.html#).

# Citation

If you feel our code or models helps in your research, kindly cite our papers:

```
@article{gluoncvnlp2020,
  author  = {Jian Guo and He He and Tong He and Leonard Lausen and Mu Li and Haibin Lin and Xingjian Shi and Chenguang Wang and Junyuan Xie and Sheng Zha and Aston Zhang and Hang Zhang and Zhi Zhang and Zhongyue Zhang and Shuai Zheng and Yi Zhu},
  title   = {GluonCV and GluonNLP: Deep Learning in Computer Vision and Natural Language Processing},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {23},
  pages   = {1-7},
  url     = {http://jmlr.org/papers/v21/19-429.html}
}

@article{he2018bag,
  title={Bag of Tricks for Image Classification with Convolutional Neural Networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  journal={arXiv preprint arXiv:1812.01187},
  year={2018}
}

@article{zhang2019bag,
  title={Bag of Freebies for Training Object Detection Neural Networks},
  author={Zhang, Zhi and He, Tong and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  journal={arXiv preprint arXiv:1902.04103},
  year={2019}
}
```
