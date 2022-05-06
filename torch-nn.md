# torch.nn
torch.nn是提供给python的一个包，包含了常用的神经网络结构。将其中的类按照功能可以划分为如下几块
```
容器
卷积层
池化层
Padding 层
非线性激活（加权和、非线性）
非线性激活（其他）
标准化层
Recurrent层
Transformer层
Linear层
Dropout层
Sparse层
距离函数
损失函数
Vision 视觉层
Shuffle 打乱层
数据并行层（多GPU、分布式）
Utilities
量化函数
Lazy Modules Initialization
```

我将按照使用顺序不断填充nn包内的的不同类的详细用法
# 容器
## 1 nn.Module 
Module属于容器，在pytorch中是所有神经网络模块的父类。
我们所创建的神经网络类应该是这个类的子类。你可以在Module中嵌套的创建Module类,如下:
```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #nn.Conv2d和nn.Conv2d都是nn.Module的子类
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```
nn.Module 有 8 个属性，都是OrderDict(有序字典)。其中比较重要的是parameters和modules属性。
`_parameters`属性里包含了该卷积层的可学习参数，这些参数的类型是 Parameter，继承自 Tensor。
`_modules`属性，本例中有序字典`_modules`会存储Model类型中nn.Module的实例如`nn.Conv2d(1, 20, 5)`

### 1总结
- 一个 module 里可包含多个子 Module。比如 Model 是一个 Module，里面包括多个卷积层、池化层、全连接层等子 module
- 一个 module 相当于一个运算，必须实现 forward() 函数
- 每个 module 都有 8 个字典管理自己的属性

除了上述的模块之外，还有一个重要的概念是模型容器 (Containers)，常用的容器有 3 个，这些容器都是继承自nn.Module
- nn.Sequetial：按照顺序包装多个网络层
- nn.ModuleList：像 python 的 list 一样包装多个网络层，可以迭代
- nn.ModuleDict：像 python 的 dict 一样包装多个网络层，通过 (key, value) 的方式为每个网络层指定名称。

### 1.1 nn.Sequential
继承自nn.Module,用来按照顺序包装多个网络层，代码实例如下
```python
import torch.nn as nn
import torch.nn.functional as F

class ModelSequetial(nn.Module):
    def __init__(self, classes): 
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2) ) 
        self.classifier = nn.Sequential( 
            nn.Linear(16*5*5, 120), 
            nn.ReLU(), 
            nn.Linear(120, 84), 
            nn.ReLU(), 
            nn.Linear(84, classes) ) 
        
        def forward(self, x): 
            x = self.features(x) 
            x = x.view(x.size()[0], -1) 
            x = self.classifier(x) 
            return x
```
在初始化时，nn.Sequetial会调用__init__()方法，将每一个子 module 添加到 自身的_modules属性中。这里可以看到，我们传入的参数可以是一个 list，或者一个 OrderDict。如果是一个 OrderDict，那么则使用 OrderDict 里的 key，否则使用数字作为 key

ModelSequetial在进行前向传播时，会进入 Model 的`forward()`函数，首先调用第一个`Sequetial`容器：`self.features`，由于`self.features`也是一个 module，因此会调用`__call__()`函数，里面调用 result = self.forward(*input, **kwargs)，进入nn.Seuqetial的forward()函数，在这里依次调用所有的 module。

notes:`__call__()`实现了之后可以让类实例被直接调用如:
```python
class Person(object):
  def __init__(self, name, gender):
    self.name = name
 
  def __call__(self, friend):
    print('My friend is %s...' % self.friend)
      
p = Person('Bob')
p(123)
```

### 1.1总结
nn.Sequetial是nn.Module的容器，用于按顺序包装一组网络层，有以下两个特性。
- 顺序性：各网络层之间严格按照顺序构建，我们在构建网络时，一定要注意前后网络层之间输入和输出数据之间的形状是否匹配
- 自带forward()函数：在nn.Sequetial的forward()函数里通过 for 循环依次读取每个网络层，执行前向传播运算。这使得我们我们构建的模型更加简洁

### 1.2 nn.ModuleList
nn.ModuleList是nn.Module的容器，用于包装一组网络层，以迭代的方式调用网络层，主要有以下 3 个方法：

- append()：在 ModuleList 后面添加网络层
- extend()：拼接两个 ModuleList
- insert()：在 ModuleList 的指定位置中插入网络层

下面的代码通过列表生成式来循环迭代创建 10 个全连接层，非常方便，只是在 forward()函数中需要手动调用每个网络层。

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = l(x)
        return x
```

### 1.3 nn.ModuleDict
nn.ModuleDict是nn.Module的容器，用于包装一组网络层，以索引的方式调用网络层，主要有以下 5 个方法：

- clear()：清空 ModuleDict
- items()：返回可迭代的键值对 (key, value)
- keys()：返回字典的所有 key
- values()：返回字典的所有 value
- pop()：返回一对键值，并从字典中删除

下面的模型创建了两个ModuleDict：self.choices和self.activations，在前向传播时通过传入对应的 key 来执行对应的网络层。
```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
```

# 卷积层
### 1.简介
1D/2D/3D 卷积
卷积有一维卷积、二维卷积、三维卷积。一般情况下，卷积核在几个维度上滑动，就是几维卷积。比如在图片上的卷积就是二维卷积。

### 2 nn.Conv2d()
这个函数的功能是对多个二维信号进行二维卷积，主要参数如下：

- in_channels：输入通道数
- out_channels：输出通道数，等价于卷积核个数
- kernel_size：卷积核尺寸
- stride：步长
- padding：填充宽度，主要是为了调整输出的特征图大小，一般把 padding 设置合适的值后，保持输入和输出的图像尺寸不变。
- dilation：空洞卷积大小，默认为 1，这时是标准卷积，常用于图像分割任务中，主要是为了提升感受野
- groups：分组卷积设置，主要是为了模型的轻量化，如在 ShuffleNet、MobileNet、SqueezeNet 中用到
- bias：偏置

