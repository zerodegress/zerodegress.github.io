---
title: 'Gradio与‘.’的文件访问问题'
description: '一个过于智障的Gradio问题'
pubDate: '2024.6.22'
heroImage: '/blog-placeholder-1.jpg'
tags: ['gradio', 'ai', 'python', 'web']
---

> 这半年我见过最智障的问题

## 开门见山

基于Gradio的应用在路径中包含哪怕一个‘.’的路径（比如说‘/home/xxxx/.local’）底下运行时，都有可能导致应用运行时错误（主要表现为网页不能正常响应），原因在于Gradio出于安全考虑禁止对绝对路径中包含‘.’的所谓的**dotfile**的访问。

> 呃，毕竟是给对安全性一无所知的人工智能开发人员开发的，所以也挺合理的罢

## 使用环境

以防万一，先声明下使用环境：

操作系统：ArchLinux（2024.6.21时保持最新）
Python版本：3.10.14,3.10.6都用过了（但是没用过conda）
显卡：Nvidia RTX3060 Laptop
Cuda版本：系统安装为12.4
PyTorch版本：自动安装的2.1.2+cu121
顺便附一下`fastfetch`：

![](/blog-images/gradio-dot-issue/fastfetch.png)

## 起因

某一天我开开心心地删掉硬盘里好久没玩的游戏腾点空间，用之前写的小工具[rerman](https://github.com/zerodegress/rerman)把Stable Diffusion Web UI克隆下来准备玩玩跑图。

吃着火锅唱着歌，突然就被麻匪给劫了！

## 遭遇错误

起初，我并不知道将rerman设定为默认会克隆到'~/.local/share/rerman/repositories'底下的某个文件夹会造成多大危害，然后就这么想当然地运行了webui，一切都看起来那么平常。

然后，安装中文化插件时就出了点岔子：当我准备好中文插件再重启webui时，我发现什么反应都没有。

思考片刻后我打算放弃汉化，转而直接跑个图先。当我点击`Generate`按钮时，一点反应也没有。啊这，这不应该吧。

于是我直接打开`f12`看看是不是有网络请求错误：
![](/blog-images/gradio-dot-issue/issue1.png)

果不其然，有一堆网络请求错误！再仔细检查一下，这些网络请求错误导致了部分脚本没有正确加载，进一步造成gradio应用不能工作。

## 尝试解决

不过我一开始还是想得太简单了，只是尝试换个姿势重新安装sdwebui，包括但不限于推倒venv重来，删了webui的数据文件。但这些努力全部木大。

第二天早上起来直接赶来实验室进一步研究一下，通过跟踪来源，查阅网络资料，问ChatGPT，暴力搜索webui项目目录等方法（以下省略一万行，你懂得），追踪到了命令行参数`--gradio-allowed-path`：这个命令行参数用于添加一个gradio允许访问的目录。于是我决定试试这个参数加上当前目录看看能不能用。

但是还是木大，问题不在这里（毕竟理论上来说gradio会自行添加自身所在目录为allowed path）。

于是又暴力搜索了半个小时，锁定到了来自`gradio`模块的脚本`routes.py`的大概364行的位置：

![](/blog-images/gradio-dot-issue/issue2.png)

哦，还有这里：

![](/blog-images/gradio-dot-issue/issue3.png)

这下就看明白了：Gradio会自动将绝对路径中带‘.’的目录加入禁止访问的目录，导致Gradio应用运行在带‘.’的目录时就会出问题。

## 最终解决方案

囫囵个移动到不带‘.’的目录呗，我还能咋办。

而且官方在我刚刚搜到的[issue](https://github.com/gradio-app/gradio/issues/5407)下的表态是：

![](/blog-images/gradio-dot-issue/issue4.png)

简单来说，他们只在`4.x`版本进行修复，而sdwebui用的`3.x`没啥办法了。

## 总结

珍爱生命，远离AI相关的开发工具（bushi

> 好歹他们认真改了，只不过依赖旧玩意的我们就惨了……