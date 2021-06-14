# Jittor-Image-Models

Jittor Image Models (`timm-jittor`) is a library for pulling together a wide variety of SOTA deep learning models in the [https://github.com/Jittor/jittor Jittor] framework. Based on timm-jittor, we achieved **the first place** of the [Dog Species Classification](https://www.educoder.net/competitions/index/Jittor-2) track in the Jittor AI Competition in 2021.

Our `timm-jittor` is modified from Py**T**orch **Im**age **M**odels (`timm`) which helps fine-tune PyTorch models list systematically by `timm` in Jittor.

More specifically, PyTorch Image Models is an excellent project created by Ross Wightman and perfected by many outstanding contributors. Details about `timm` is available at: https://github.com/rwightman/pytorch-image-models  

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. More details about Jittor can be found via: https://github.com/Jittor/jittor  

In our `timm-jittor`, we reproduce part of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) in the [Jittor](https://github.com/Jittor/jittor) deep learning framework, and also provide a training demo to make it easier for you to get started.

## Update News

### June 10, 2021
* Add HRNet models.

## Currently Supported Models
* DeiT (Vision Transformer) - https://arxiv.org/abs/2012.12877
* EfficientNet
    * EfficientNet (B0-B7) - https://arxiv.org/abs/1905.11946
    * EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665
    * EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
* HRNet - https://arxiv.org/abs/1908.07919
* ResNet/ResNeXt
    * ResNet (v1b/v1.5) - https://arxiv.org/abs/1512.03385
    * ResNeXt - https://arxiv.org/abs/1611.05431
* ViT - https://arxiv.org/abs/2010.11929

More models provided by timm will continue to be updated.

## Results
Model validation results can be found in the following url: https://rwightman.github.io/pytorch-image-models/results/  

## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Xuhao Sun: sun_xu_hao@163.com

Yang Shen: shenyang_98@njust.edu.cn

[Xiu-Shen Wei](http://www.weixiushen.com/) (Primary contact): weixs.gm@gmail.com
