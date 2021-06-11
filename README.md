# Jittor-Image-Models


Pytorch-image-models is an excellent project created by Ross Wightman and perfected by many outstanding contributors. More about pytorch-image-models: https://github.com/rwightman/pytorch-image-models    
Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. More about Jittor deep learning framework: https://github.com/Jittor/jittor  

We reproduced part of pytorch-image-models(mainly timm) in Jittor deep learning framework. By using timm-jittor, you can easly use pytorch models provided by timm and fine-tune on Jittor. We give one training demo which may help you get started faster.  

## Now Supported Models
* EfficientNet
    * EfficientNet (B0-B7) - https://arxiv.org/abs/1905.11946
    * EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
    * EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665
* ResNet/ResNeXt
    * ResNet (v1b/v1.5) - https://arxiv.org/abs/1512.03385
    * ResNeXt - https://arxiv.org/abs/1611.05431
* ViT - https://arxiv.org/abs/2010.11929
* DeiT (Vision Transformer) - https://arxiv.org/abs/2012.12877
* HRNet - https://arxiv.org/abs/1908.07919

