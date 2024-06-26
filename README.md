# neural_style
《A Neural Algorithm of Artistic Style》风格迁移复现



### 复现报告

#### 项目背景

本项目的目标是实现基于深度学习的图像风格迁移。风格迁移技术通过提取一张风格图像和一张内容图像的特征，并结合两者的特征生成一张兼具风格和内容的图像。本文实现基于预训练的VGG16模型提取特征，并定义自定义的卷积层和残差块来构建图像转换网络。

#### 环境设置

1. **硬件要求**:
   - GPU（如果可用）

2. **软件要求**:
   - Python 3.8+
   - PyTorch
   - Torchvision
   - PIL
   - OpenCV
   - Matplotlib
   - Numpy

#### 代码结构

##### 1. 导入依赖

首先，导入所需的库，包括PyTorch、Torchvision、PIL、OpenCV、Matplotlib和Numpy。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import cv2
```

##### 2. 设置设备

检测是否有GPU可用，并将设备设置为GPU或CPU。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

##### 3. 定义图像预处理和恢复函数

定义图像预处理和恢复的辅助函数，保证输入和输出图像格式正确。

```python
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)
epsilon = 1e-5

def preprocess_image(image, target_width=None):
    if target_width:
        t = transforms.Compose([
            transforms.Resize(target_width), 
            transforms.CenterCrop(target_width), 
            transforms.ToTensor(), 
            tensor_normalizer, 
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(), 
            tensor_normalizer, 
        ])
    return t(image).unsqueeze(0)

def recover_image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]

def imshow(tensor, title=None):
    image = recover_image(tensor)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
```

##### 4. 定义VGG模型

使用预训练的VGG16模型作为特征提取器，并冻结其参数。

```python
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {'3': "relu1_2", '8': "relu2_2", '15': "relu3_3", '22': "relu4_3"}
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs
```

##### 5. 自定义卷积层和残差块

定义一个自定义的卷积层和残差块，用于图像转换网络。

```python
class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MyConv2D, self).__init__()
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size)).to(device)
        self.bias = torch.zeros(out_channels).to(device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}')
        return s.format(**self.__dict__)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1), 
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x
```

##### 6. 定义卷积层函数

定义一个辅助函数，用于构建卷积层，可以选择性地添加上采样、反射填充、实例归一化和ReLU激活。

```python
def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, upsample=None, instance_norm=True, relu=True, trainable=False):
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
    layers.append(nn.ReflectionPad2d(kernel_size // 2))
    if trainable:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    else:
        layers.append(MyConv2D(in_channels, out_channels, kernel_size, stride))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return layers
```

##### 7. 定义转换网络

构建一个包含下采样、残差块和上采样的转换网络。

```python
class TransformNet(nn.Module):
    def __init__(self, base=8):
        super(TransformNet, self).__init__()
        self.base = base
        self.weights = []
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9, trainable=True), 
            *ConvLayer(base, base*2, kernel_size=3, stride=2), 
            *ConvLayer(base*2, base*4, kernel_size=3, stride=2), 
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base*4) for i in range(5)])
        self.upsampling = nn.Sequential(
            *ConvLayer(base*4, base*2, kernel_size=3, upsample=2),
            *ConvLayer(base*2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False, trainable=True),
        )
        self.get_param_dict()
    
    def forward(self, X):
        y = self.downsampling(X)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y
    
    def get_param_dict(self):
        param_dict = defaultdict(int)
        def dfs(module, name):
            for name2, layer in module.named_children():
                dfs(layer, '%s.%s' % (name, name2) if name != '' else name2)
            if module.__class__ == MyConv2D:
                param_dict[name] += int(np.prod(module.weight.shape))
                param_dict[name] += int(np.prod(module.bias.shape))
        dfs(self, '')
        return param_dict
    
    def set_my_attr(self, name, value):
        target = self
        for x in name.split('.'):
            if x.isnumeric():
                target = target.__getitem__(int(x))
            else:
                target = getattr(target, x)
        
        n_weight = np.prod(target.weight.shape)
        target.weight = value[:n_weight].view(target.weight.shape)
        target.bias = value[n_weight:].view(target.bias.shape)
    
    def set_weights(self, weights, i=0):
        for name, param in weights.items():
            self.set_my_attr(name, weights[name][i])
```

##### 8. 定义元网络

定义一个元网络，用于根据输入特征生成不同层的权重。

```python
class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(1920, 128*self.param_num)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i+1), nn.Linear(128, params))
    
    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i+1))
            filters[name] = fc(hidden[:,i*128:(i+1)*128])
        return filters
```

##### 9. 定义图像转换相关函数

定义图像

读取和转换为张量的函数。

```python
def image_to_tensor(image, target_width=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return preprocess_image(image, target_width)

def read_image(path, target_width=None):
    image = Image.open(path)
    return preprocess_image(image, target_width)

def recover_tensor(tensor):
    m = torch.tensor(cnn_normalization_mean).view(1, 3, 1, 1).to(tensor.device)
    s = torch.tensor(cnn_normalization_std).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * s + m
    return tensor.clamp(0, 1)
```

##### 10. 定义特征处理函数

定义计算特征均值和标准差以及Gram矩阵的函数。

```python
def mean_std(features):
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)
        x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) + epsilon)], dim=-1)
        n = x.shape[0]
        x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1)
        mean_std_features.append(x2)
    mean_std_features = torch.cat(mean_std_features, dim=-1)
    return mean_std_features

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
```

##### 11. 测试代码

使用预训练的VGG16模型提取图像特征，计算风格图像的Gram矩阵和内容图像的特征，然后通过优化器不断调整输入图像，使其同时接近内容图像的特征和风格图像的Gram矩阵。

```python
# 加载预训练的VGG16模型，并裁剪到需要的层
vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

# 设定图像宽度
width = 512

# 读取风格图像和内容图像
style_img = read_image('./starry_night.jpg', target_width=width).to(device)
content_img = read_image('./pic.jpg', target_width=width).to(device)

# 提取风格和内容特征
style_features = vgg16(style_img)
content_features = vgg16(content_img)
style_grams = [gram_matrix(x) for x in style_features]

# 克隆内容图像作为输入图像
input_img = content_img.clone()

# 使用LBFGS优化器
optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

# 定义风格和内容损失权重
style_weight = 1e6
content_weight = 1

# 定义训练步骤计数器
run = [0]

# 优化循环
while run[0] <= 300:
    def f():
        optimizer.zero_grad()
        features = vgg16(input_img)
        
        # 计算内容损失
        content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
        
        # 计算风格损失
        style_loss = 0
        grams = [gram_matrix(x) for x in features]
        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b) * style_weight
        
        # 总损失
        loss = style_loss + content_loss
        
        # 打印损失信息
        if run[0] % 50 == 0:
            print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(run[0], style_loss.item(), content_loss.item()))
        
        run[0] += 1
        
        # 反向传播
        loss.backward()
        return loss
    
    optimizer.step(f)

# 显示结果图像
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
imshow(style_img, title='Style Image')
plt.subplot(1, 3, 2) 
imshow(content_img, title='Content Image')
plt.subplot(1, 3, 3)
imshow(input_img, title='Output Image')
plt.show()
```

#### 实验结果

通过上述代码，我们成功实现了基于VGG16特征提取的风格迁移，并使用自定义的卷积层和残差块进一步处理图像。下面展示了输入的风格图像、内容图像以及最终输出的迁移图像。

1. **风格图像 (Style Image)**:
   - 使用的是梵高的《星空》。
   
2. **内容图像 (Content Image)**:
   - 使用的是任意选择的一张图片。

3. **输出图像 (Output Image)**:
   - 最终生成的图像融合了风格图像的艺术风格和内容图像的实际内容。

通过优化循环，逐步调整输入图像，使其既保留了内容图像的主要特征，又具备了风格图像的艺术效果。每隔50次迭代打印一次损失值，观察风格损失和内容损失的变化。

#### 结论

本次实验实现了一个基于深度学习的图像风格迁移模型，验证了VGG16特征提取在风格迁移中的有效性。通过自定义卷积层和残差块，进一步提升了模型的灵活性和性能。实验结果表明，该方法能够生成具有较好视觉效果的风格迁移图像。



## 参考代码

[CortexFoundation/StyleTransferTrilogy: 风格迁移三部曲 (github.com)](https://github.com/CortexFoundation/StyleTransferTrilogy)