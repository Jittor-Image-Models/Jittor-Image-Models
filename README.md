# Jittor Image Models

**J**ittor **Im**age **M**odels (`jimm`) is a library for pulling together a wide variety of SOTA deep learning models in the [Jittor](https://github.com/Jittor/jittor) framework. Based on `jimm`, we achieved **the first place** of the [Dog Species Classification](https://www.educoder.net/competitions/index/Jittor-2) track in the Jittor AI Competition in 2021.

Our `jimm` is modified from Py**T**orch **Im**age **M**odels (`timm`) which helps fine-tune PyTorch models list systematically by `timm` in Jittor.

More specifically, PyTorch Image Models (`timm`) is an excellent project created by Ross Wightman and perfected by many outstanding contributors. Details about `timm` is available at: https://github.com/rwightman/pytorch-image-models  

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. More details about Jittor can be found via: https://github.com/Jittor/jittor  

In our `jimm`, we reproduce part of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) in the [Jittor](https://github.com/Jittor/jittor) deep learning framework, and also provide a training demo to make it easier for you to get started.

## Update News
### Feb 25, 2022
* Add VAN, VAN pretrained models can be download from https://github.com/Visual-Attention-Network/VAN-Classification.
* You have to transfer the download .pth file by following:
* model = torch.load('van_base_828.pth', map_location=torch.device('cpu'))
* torch.save(model['state_dict'],'van_base.pth')

### Sep 06, 2021
* Add VOLO, Swin Transformer, EfficientNet-V2
* VOLO pretrained models can be download from https://github.com/sail-sg/volo.

### June 10, 2021
* Add HRNet models.

## Currently Supported Models
* DeiT (Vision Transformer) - https://arxiv.org/abs/2012.12877
* EfficientNet
    * EfficientNet (B0-B7) - https://arxiv.org/abs/1905.11946
    * EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665
    * EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
    * EfficientNet V2 - https://arxiv.org/abs/2104.00298
* HRNet - https://arxiv.org/abs/1908.07919
* ResNet/ResNeXt
    * ResNet (v1b/v1.5) - https://arxiv.org/abs/1512.03385
    * ResNeXt - https://arxiv.org/abs/1611.05431
    * Weakly-supervised (WSL) Instagram pretrained / ImageNet tuned ResNeXt101 - https://arxiv.org/abs/1805.00932
    * Semi-supervised (SSL) / Semi-weakly Supervised (SWSL) ResNet/ResNeXts - https://arxiv.org/abs/1905.00546
* Swin Transformer - https://arxiv.org/abs/2103.14030
* VAN - https://arxiv.org/abs/2202.09741
* ViT - https://arxiv.org/abs/2010.11929
* VOLO - https://arxiv.org/abs/2106.13112

More models provided by timm will continue to be updated.

## Results
Model validation results can be found in the following url: https://rwightman.github.io/pytorch-image-models/results/  

## TODO List
- [ ] RepVGG - https://arxiv.org/abs/2101.03697
- [ ] Big Transfer ResNetV2 (BiT) - https://arxiv.org/abs/1912.11370
- [ ] NFNet-F - https://arxiv.org/abs/2102.06171

## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Xuhao Sun: sunxh@njust.edu.cn

Yang Shen: shenyang_98@njust.edu.cn

[Xiu-Shen Wei](http://www.weixiushen.com/) (Primary contact): weixs.gm@gmail.com
