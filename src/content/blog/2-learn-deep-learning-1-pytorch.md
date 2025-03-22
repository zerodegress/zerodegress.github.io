---
title: '学习深度学习（1）Pytorch'
description: '深度学习入门'
pubDate: '2025-03-19'
heroImage: '/blog-placeholder-2.jpg'
tags: ['deep-learning', 'pytorch', 'python']
---

> 本文章一定程度上基于[https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html]编写

要学习深度学习，自然是要先学习一个能用的深度学习框架，为了简单我们选择`pytorch`：
这是一个基于`Python`的框架，地位类似于`numpy`，但是不同之处在于有先进的`自动求导`
和在显卡、`NPU`上运行计算程度的能力。
为什么不选择`Rust`或者`C++`上有的深度学习框架呢，
因为好写（还可以写`Jupyter Notebook`），用这俩玩意我还得编译半天。

## 张量

机器学习中，无论输出还是输入都是一坨数据，通常来说我们会用n维数组来表示这个数据方便计算和展示。
比如说黑白图像可以看成是一个二维数组，其中每一个数据表示对应点位的灰度，如果是彩色图像那便是三维数组，
原来的数改成三元组。

在众所周知的`numpy`中，n为数组被称呼为`ndarray`，但这个名称已经遭受了过度侵犯，因此在`pytorch`中我们
我们给它改名叫`Tensor`（张量），但很显然在机器学习盛行的当下这个名字又双被过度侵犯了，以至于和物理学
中的张量搞混，为了保险起见我事先声明一下，深度学习中`Tensor`真的就是单纯的一坨数据，而不是物理学那堆玩意。

同任何正常的`python`库一样，`pytorch`需要导入使用:

```python
import torch
import numpy
```

> 另外提一嘴，其实这家伙本来就叫`torch`，并且最初也不是给`python`使用的，是给`lua`用的
> 不过很显然`lua`没有赶上这波机器学习风潮惨遭冷落，由此可见小语言真的是没人权

### 初始化

张量可以通过多种方式进行初始化，就像你在`numpy`中使用的一样：

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

> 数据类型会自动推断出来。如果你不确定是什么类型的，可以调用`x_data.dtype`看到葫芦里面装了什么药

也可以从`numpy`中的`ndarray`中初始化

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

> 方便和那些本来是和`numpy`一起用的家伙一起用

另外还可以从另一个张量初始化，这里略去。

`torch`还有以下几个方法可以创建张量：

- `torch.rand`: 随机生成
- `torch.ones`：全是1
- `torch.zeros`：全是0

剩下的自己看看文档研究吧。

### 属性

张量除了里面包裹的数据还有一些额外的属性，比较常用的有：

- `shape`：形状，大概描述了你的张量每层是几维的有几个元素，例如`(2,2)`就表示一个2x2的矩阵。
- `dtype`：数据类型，比如浮点数或者整数
- `device`：在哪个设备上存储着，没指定的话默认存到`cpu`上

### 操作

太多了，简单说几个重点：

- `tensor.to`：把张量转移到指定设备，比如从你的`cpu`中转移到显卡中
- `tensor[x,y]`：手感类似`numpy`，不多解释
- `torch.cat`：把张量沿给定维度连接起来
- `tensor.mul（tensor1 * tensor2）`：两种写法都可，对应位置元素相乘组成新的张量
- `tensor.matmul`：矩阵乘法，叉乘
- `tensor.xxx_`：一系列就地操作，操作完后只会改变自己而不是返回一个新的张量，可以节省存储空间

### numpy兼容

- `torch.from_numpy`：从`numpy`到`pytorch`
- `tensor.numpy`：从`pytorch`到`numpy`

另外，操作后原先的值还能用，并且会和转换后的值互通，方便兼容了属于是

## 自动求导

`pytorch`一大有别于`numpy`的地方在于自动求导，这让它可以自动的训练神经网络而不用你手动去调整参数

### 背景

神经网络(NN)是对某些输入数据执行的嵌套函数的集合，这些函数由参数（权重和偏差）定义，这些参数在`pytorch`中的具象化体现就是张量。

训练神经网络统共分两步:

1. 前向传播：也就是运行，这样神经网络就能对输入做出推测，也就是得到输出
2. 反向传播：也就是调参，通过输出与真值的差异对于函数参数梯度的导数来递归向下优化参数

然后就没了，就这么简单，感谢`pytorch`强大的功能吧，我们只要明白个大概，然后喂数据进去就行了。

### 在Pytorch中使用

我们拿`resnet`先来开刀，另外这段教程不保证gpu下有用，所以别试cuda了：

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
# 加载 resnet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 生成随机数据，维度分别代表：批次大小，通道，高度，宽度
data = torch.rand(1, 3, 64, 64)
# 生成随机标签，维度分别代表：批次大小，标签数量
labels = torch.rand(1, 1000)

# 正向传播
prediction = model(data)

# 计算损失，这是一种非常简单的计算方法
loss = (prediction - labels).sum()
# 反向传播
loss.backward() # backward pass

# 选取一个优化器
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# 应用递归向下优化
optim.step() #gradient descent
```

就这样，一次简单的训练完成了

## 神经网络

首先声明一下，**神经网络**跟生物学上的同名名词毫无关系，最多是结构上容易联想到一起。

![卷积神经网络](/blog-images/2-learning-deeplearning-1-pytorch/num_class_network.png)

上图是一个简单的前馈网络，它接受输入，一个接一个地通过几层馈送，最后给出输出。

神经网络的典型训练过程如下：

- 定义具有一些可学习参数（或权重）的神经网络
- 迭代输入数据集
- 通过网络处理输入
- 计算损失（输出和真值的间距）
- 将梯度传播回网络的参数
- 更新网络的权重，通常使用金蛋的更新规则：`权重 = 权重 - 学习率 * 梯度`

### 定义神经网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1输入图像通道, 6输出通道, 5x5方形卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射变换操作：y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入维度中的5*5来自图像尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # 卷积层C1：1个输入通道，6个输出通道，5x5方形卷积
        # 使用ReLU激活函数，输出尺寸为(N, 6, 28, 28)张量，N为批次大小
        c1 = F.relu(self.conv1(input))
        # 下采样层S2：2x2网格，纯功能层
        # 该层无参数，输出(N, 6, 14, 14)张量
        s2 = F.max_pool2d(c1, (2, 2))
        # 卷积层C3：6个输入通道，16个输出通道，5x5方形卷积
        # 使用ReLU激活函数，输出(N, 16, 10, 10)张量
        c3 = F.relu(self.conv2(s2))
        # 下采样层S4：2x2网格，纯功能层
        # 该层无参数，输出(N, 16, 5, 5)张量
        s4 = F.max_pool2d(c3, 2)
        # 展平操作：纯功能层，输出(N, 400)张量
        s4 = torch.flatten(s4, 1)
        # 全连接层F5：输入(N, 400)张量
        # 输出(N, 120)张量，使用ReLU激活函数
        f5 = F.relu(self.fc1(s4))
        # 全连接层F6：输入(N, 120)张量
        # 输出(N, 84)张量，使用ReLU激活函数
        f6 = F.relu(self.fc2(f5))
        # 输出层：输入(N, 84)张量
        # 输出(N, 10)张量
        output = self.fc3(f6)
        return output


net = Net()
print(net)
```

你只需要定义 `forward` 函数，`backward` 函数（计算梯度的地方）就会使用 autograd 自动为你定义。你可以在 `forward` 函数中使用任何张量操作。

模型的可学习参数由 `net.parameters()` 返回

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

让我们尝试一个随机的 32x32 输入。注意：这个网络 （LeNet） 的预期输入大小为 32x32。要在 MNIST 数据集上使用此网络，请将数据集中的图像大小调整为 32x32。

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

使用随机渐变将所有参数和反向传播的渐变缓冲区归零：

```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

对这段感觉不知所措？没事，接下来我们就不会用土办法反向传播了，那是**损失函数**该干的

### 损失函数

损失函数输入一对输出和真值，并计算一个值，估计输出与真值的距离。

`torch.nn`包下有几种不同的损失函数，最简单的一种是`nn.MSELoss`，它计算输出与真值之间的均方误差。

比如说：

```python
output = net(input)
target = torch.randn(10)  # 随便举例的
target = target.view(1, -1)  # 确保与输出有相同形状
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

`loss`里面包裹着误差和用于反向传播的东西。

### 反向传播

要反向传播误差，就是调用`loss.backward()`了。但在此之前需要清除现有的梯度，以防梯度累积到现有的梯度中。

```python
net.zero_grad()     # 清除所有参数的梯度缓冲

print('conv1.bias.grad 反向传播前')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad 反向传播后')
print(net.conv1.bias.grad)
```

很好，你已经学会损失函数怎么用了，接下来是更新权重。

### 更新权重

更新权重后模型训练才正式完成。

实践中最简单的更新规则是随机梯度下降（Stochastic Gradient Descent (SGD)）：

```
weight = weight - learning_rate * gradient
```

对其最简单的`Python`实现是：

```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

不过使用神经网络时事情可能会更麻烦一些，为了省事请使用`torch.optim`来解决。

比如说：

```python
import torch.optim as optim

# 创建你的优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在你的训练循环中
optimizer.zero_grad()   # 无需多言
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新权重
```

你已经学会使用优化器来更新权重了，接下来我们来真格的。

## 训练分类器

分类器是一种能将数据分成若干类型的神经网络，我们选择它来作为我们的第一个实践项目。

### 数据是什么？

通常来说是图像、文本、音频、视频，我们需要给处理成张量后才能用于训练和推理。

### 数据从哪里来？

网络爬虫，或者自己收集。

### 数据到哪里去？

当然是用于训练和验证模型，然后就可以用于一般数据的推理了。

### 训练图像分类器

幸运的是，网上有现成的图像数据集供我们训练使用，所以我们可以省事了。

我们要进行如下步骤：

1. 使用`torchvision`加载和规范化 CIFAR10 训练和测试数据集。
2. 定义卷积神经网络
3. 定义损失函数
4. 在训练数据上训练网络
5. 在测试数据上测试网络

#### 1.加载并规范化 CIFAR10

`torchvision`提供的工具很不错，可以实现数据集拉取、整合、训练一条龙。

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
  root='./data', 
  train=True,
  download=True, 
  transform=transform
)
trainloader = torch.utils.data.DataLoader(
  trainset, 
  batch_size=batch_size,
  shuffle=True, 
  num_workers=2
)

testset = torchvision.datasets.CIFAR10(
  root='./data', 
  train=False,
  download=True, 
  transform=transform
)
testloader = torch.utils.data.DataLoader(
  testset, 
  batch_size=batch_size,
  shuffle=False, 
  num_workers=2
)

classes = (
  'plane', 
  'car', 
  'bird', 
  'cat',
  'deer', 
  'dog', 
  'frog', 
  'horse', 
  'ship', 
  'truck'
)
```

预览训练图像

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # 未归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机选几张
dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
# 打印对应的标签
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
```

#### 2.定义卷积神经网络

改改之前的就能用了

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 平铺除了批次数的所有维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

#### 3.定义损失函数和优化器

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

#### 4.训练神经网络

```python
for epoch in range(2):  # 两轮训练

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入，输入格式为 [inputs, labels]
        inputs, labels = data

        # 归零
        optimizer.zero_grad()

        # 前向传播，反向传播，更新权重
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] 损失: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('训练完成')
```

很简单吧，现在来保存模型

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

#### 5.在测试数据上测试神经网络

先看一眼图像

```python
dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
```

如果你不想再训练一遍了，我们可以加载之前训练好的参数

```python
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
```

现在问问模型的想法

```python
outputs = net(images)
```

输出是是个类型的分数，分数越高意味着模型认为图像属于特定类可能性越高：

```python
_, predicted = torch.max(outputs, 1)

print(
  'Predicted: ', ' '.join(
    f'{classes[predicted[j]]:5s}' for j in range(4)
  )
)
```

然后对测试集全部试验一遍

```python
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

如果你的输出准确率高于`10%`，那说明你的模型差强人意，起码比瞎蒙准不少

### 在GPU上训练模型

在GPU上训练模型非常简单，你只需要先获取下可用的`cuda`设备：

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

> 我手头没`Rocm`或者什么`npu`，其他的自己试吧

然后把神经网络转换到设备上：

```python
net.to(device)
```

最后还要记得把训练验证测试数据统统发送到设备上：

```python
inputs, labels = data[0].to(device), data[1].to(device)
```

## 小结

大功告成，你现在已经学会神经网络的训练和应用了，快去开发一款爆款AI应用吧