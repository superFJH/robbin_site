# Mobile-deep-learning（MDL）

![License](https://img.shields.io/github/license/simpeg/simpeg.svg) ![Travis](https://img.shields.io/travis/rust-lang/rust.svg)

#### Free and open source mobile deep learning framework, deploying by Baidu.

This research aims at simply deploying CNN on mobile devices, with low complexity and high speed.
It supports calculation on iOS GPU, and is already adopted by Baidu APP.

* Size: 340k+ (on arm v7)
* Speed: 40ms (for iOS Metal GPU Mobilenet) or 30 ms (for Squeezenet)


百度研发的移动端深度学习框架，致力于让卷积神经网络极度简单的部署在手机端。目前正在手机百度内运行。支持iOS gpu计算。体积小，速度快。

* 体积 armv7 340k+
* 速度 iOS GPU mobilenet 可以达到 40ms、squeezenet 可以达到 30ms

## Getting Started
#### Showcase

我在原来的mdl的框架下加入了ssd所需要的一些层，并与caffe做了交叉调试，确认每一层的运行与caffe一致，但是目前存在着mdl转化的int8参数在卷级中结果与caffe不一致

我之后会陆续加入相关支持，改进代码。
