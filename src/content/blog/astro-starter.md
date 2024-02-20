---
title: 'Astro：起步'
description: '从Astro起步，一步一脚印'
pubDate: '2024.2.20'
heroImage: '/blog-placeholder-1.jpg'
---

[Astro](https://astro.build/)是一个用于生成文档、静态博客等内容驱动的静态网站生成工具。本站点就是使用Astro生成，并在GitHub Pages上进行发布。

## 开始

在Astro官网上可以找到很多已有的网站模板，我选择的是[博客模板](https://github.com/withastro/astro/tree/latest/examples/blog)。

随便找一处空地，输入以下命令来通过这个模板新建一个项目：

```sh
pnpm create astro@latest --template blog
```

生成步骤和`create-vite`和`create-react-app`什么的差不多，在此省略。在此使用`TypeScript`，并选择了最严格的那款配置文件。

接下来用`VSCode`打开项目目录。不过这时候还不能开始作业，因为相关依赖还没有安装完毕，输入以下命令安装依赖：

```sh
pnpm i
```

个人习惯额外安装[ESLint](https://eslint.org)和[Prettier](https://prettier.io)并配置好相关插件，不过其实没必要。

## 工作空间布局

我们再把目光调转回工作空间，认识一下astro项目下的各个文件夹。

- `components` 文件夹下放置着常用的`Astro`组件。
- `content` 文件夹下放置的是Astro的各种Markdown或其他内容文件，比如说我们的博文会放置在其下的`blog`文件夹中，包括`MD`和`MDX`格式的文件。
- `layouts` 文件夹不是很重要，不过习惯上用这个文件夹来存储一些模板网页布局，比如说我们博文的格式。
- `pages` 文件夹下放置着网页的路由布局，每一个`.astro`文件或者文件夹下的`index.astro`都对应着一个路由。**特别的**，像`[...somename].astro`这样的名称意味着在该文件所在路由下有不确定数量的类似页面，当然`index.astro`除外。
- `styles` 文件夹下习惯上用来放置网页`CSS`样式。
- `.astro` 文件夹下放置着由`Astro`自动生成的辅助开发用文件，不要动就完事了。

## Astro组件

`.astro`文件都是Astro组件，是一种模板组件，通常上半部分可以写`JS`，下半部分写类似`JSX`的网页布局（当然也可以写`TypeScript`）。

Astro组件的写法跟`Vue`组件比较像，也可以从一个Astro组件中引入其他Astro组件，如果配置好了那么还可以引入`React`，`Vue`等组件。

每个Astro组件还会被暴露一个`Astro`变量，可以通过`Astro.props`获取一些有用的关于本页面的信息，比如说markdown文件或者是markdown的描述什么的。

## astro.config.js

> 为什么他们要用`mjs`做后缀？太别扭了。所以我改成`js`做后缀，毕竟都4202年了谁还写`CommonJS`啊，不会吧不会吧，不会真有人还在`cjs`上踌躇不前吧

`astro.config.js`是整个Astro项目的配置文件，可以在内部配置Astro项目相关的配置。

比如说加入React支持，然后接着在Astro里面堆React。

## 小结

总的来说，Astro没有什么特别复杂的概念，只要知道你该在哪里写你的代码就好了。