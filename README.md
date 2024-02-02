# Usage Instructions

In ntrain.py, you can use the -merge parameter to employ the batch mix method for acceleration during training. When the merge parameter is greater than 1, the code will perform acceleration. However, not all models support acceleration, only models with mixt_ in their names do. Otherwise, an error may occur due to inconsistency in the forms of outputs and labels.






## Training Models

Here are examples of training using the command line:

### 1. Using the c14mixt_inceptionv3 model, enabling GPU acceleration, setting the merge parameter to 2, and a learning rate of 0.04:

```bash
python ntrain.py -net c14mixt_inceptionv3 -gpu -merge 2 -lr 0.04
```

### 2. Using the c14mixt_inceptionv3 model, enabling GPU acceleration, setting the merge parameter to 2, a learning rate of 0.04, and enabling mixed precision training:
```bash
python ntrain.py -net c14mixt_inceptionv3 -gpu -merge 2 -lr 0.04 -fp16
```

### 3. Using the resnet18 model, enabling GPU acceleration, and a learning rate of 0.04:

```bash
python ntrain.py -net resnet18 -gpu -lr 0.04
```




Model List
Here is a list of available models:

c2mixt_resnet18
c2mixt_resnet34
c2mixt_resnet50
c2mixt_resnet101
c2mixt_resnet152
densenet121
densenet161
densenet169
densenet201
googlenet
inceptionv3
inceptionv4
inceptionresnetv2
xception
resnet18
resnet34
resnet50
resnet101
resnet152
preactresnet18
preactresnet34
preactresnet50
preactresnet101
preactresnet152
resnext50
resnext101
resnext152
shufflenet
shufflenetv2
squeezenet
mobilenet
mobilenetv2
nasnet
attention56
attention92
seresnet18
seresnet34
seresnet50
seresnet101
seresnet152
wideresnet
stochasticdepth18
stochasticdepth34
stochasticdepth50
stochasticdepth101
c4mixt_resnet18
c4mixt4_resnet34
c4mixt4_resnet50
c4mixt4_resnet101
c4mixt_resnet152
c2c4mixt_resnet18
c2c4mixt_resnet34
c2c4mixt_resnet50
c2c4mixt_resnet101
c2c4mixt_resnet152
c14mixt_inceptionv3
c6mixt_inceptionv3
c3mixt_inceptionv3
c3mixt_resnet18
c3mixt_resnet34
c3mixt_resnet50
c3mixt_resnet101
c3mixt_resnet152



可以使用加速的有
c2mixt_resnet18
c2mixt_resnet34
c2mixt_resnet50
c2mixt_resnet101
c2mixt_resnet152
c4mixt_resnet18
c4mixt4_resnet34
c4mixt4_resnet50
c4mixt4_resnet101
c4mixt_resnet152
c2c4mixt_resnet18
c2c4mixt_resnet34
c2c4mixt_resnet50
c2c4mixt_resnet101
c2c4mixt_resnet152
c14mixt_inceptionv3
c6mixt_inceptionv3
c3mixt_inceptionv3
c3mixt_resnet18
c3mixt_resnet34
c3mixt_resnet50
c3mixt_resnet101
c3mixt_resnet152
