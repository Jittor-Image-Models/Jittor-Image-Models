# Jittor-Image-Models

Jittor Image Models (`timm-jittor`) is a project modified form Py**T**orch **Im**age **M**odels (`timm`) which helps fine-tune pytorch models list systematically by timm in the Jittor deep learning framework. 

PyTorch Image Models is an excellent project created by Ross Wightman and perfected by many outstanding contributors. More about pytorch-image-models: https://github.com/rwightman/pytorch-image-models    
Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. More about Jittor deep learning framework: https://github.com/Jittor/jittor  

We reproduce part of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) in the [Jittor](https://github.com/Jittor/jittor) deep learning framework. We provide a training demo to make it easier for you to get started.  

## Now Supported Models
* DeiT (Vision Transformer) - https://arxiv.org/abs/2012.12877
* EfficientNet
    * EfficientNet (B0-B7) - https://arxiv.org/abs/1905.11946
    * EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
    * EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665
* HRNet - https://arxiv.org/abs/1908.07919
* ResNet/ResNeXt
    * ResNet (v1b/v1.5) - https://arxiv.org/abs/1512.03385
    * ResNeXt - https://arxiv.org/abs/1611.05431
* ViT - https://arxiv.org/abs/2010.11929

## Results
Model validation results can be found in the following url: https://rwightman.github.io/pytorch-image-models/results/  
Validation results in the Jittor deep learning framework will be updated in the future.
