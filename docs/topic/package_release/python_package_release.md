# 简化安装与发包

## 问题：

1. [如何去除 ubuntu 的 apt 安装依赖？](#conda-代替系统依赖)
2. [如何支持普通用户和开发者两种安装的需求，尽量减少普通用户所需的依赖？](#区分install模式和develop模式)
3. [如何进行 python 包的动态安装？](#python-包的动态安装)
4. [如何进行 python 项目编包？](#python-编包方法)
5. [发包前要有什么准备？](#关于发包前的准备工作)
6. [发 C++ 包需要注意的东西？](#manylinux)


## conda 代替系统依赖

conda 可以用来代替一些 apt-get 安装的系统依赖，这样可以让项目适用于除了 ubuntu 以外的系统。

使用 conda 可以安装 sox、 libsndfile、swig 等 paddlespeech 需要的依赖：

```bash
conda install -y -c conda-forge sox libsndfile
```

部分系统会缺少 libbzip2 库，这个 paddlespeech 也是需要的，这也可以用 conda 安装：

```bash
conda install -y -c bzip2
```

conda 也可以安装 linux 的 C++ 的依赖：

```bash
conda install -y -c gcc_linux-64=8.4.0 gxx_linux-64=8.4.0
```

#### 剩余问题：使用 conda 环境编译 kenlm 失败。目前在 conda 环境下编译 kenlm 会出现链接失败的问题

目前知道需要的依赖：

```bash
conda install -c conda-forge eigen boost cmake
```

## 区分install模式和develop模式

可以在 setup.py 中划分 install 的依赖（基本依赖）和 develop 的依赖 （开发者额外依赖）。 setup_info 中 `install_requires` 设置 install 的依赖，而在 `extras_require` 中设置 `develop` key 为 develop 的依赖。
普通安装可以使用：

```bash
pip install . 
```

另外使用 pip 安装已发的包也是使用普通安装的：

```
pip install paddlespeech
```

而开发者可以使用如下方式安装，这样不仅会安装 install 的依赖，也会安装 develop 的依赖， 即：最后安装的依赖 = install 依赖 + develop 依赖：

```bash
pip install -e .[develop]
```

## python 包的动态安装

可以使用 pip 包来实现动态安装：

```python
import pip
if int(pip.__version__.split('.')[0]) > 9:
        from pip._internal import main
    else:
        from pip import main
    main(['install', package_name])
```

## python 编包方法

#### 创建 pypi的账号

创建 pypi 账号

#### 下载 twine

```
pip install twine
```

#### python 编包

编写好 python 包的 setup.py, 然后使用如下命令编 wheel 包：

```bash
python setup.py bdist_wheel
```

如果要编源码包，用如下命令：

```bash
python setup.py sdist
```

#### 上传包

```bash
twine upload dist/wheel包
```

输入账号和密码后就可以上传 wheel 包了

#### 关于python 包的发包信息

主要可以参考这个[文档](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/?highlight=find_packages)


## 关于发包前的准备工作

#### 拉分支
在发包之前需要拉分支。例如需要发 0.1.0 版本的正式包，则需要拉一个 r0.1 的分支。并且在这个 r0.1 分支的包上面打 0.1.0 的tag。在拉分支之前可以选择性的使用 rc 版本发一个正式版前的试用包，例如0.1.0rc0，等到rc包测试通过后，再拉分支（如果是发 0.1.1 包，则 merge r0.1分支），打tag，完成发包。

关于打tag的命令，可以参考[git 基础](https://git-scm.com/book/zh/v2/Git-%E5%9F%BA%E7%A1%80-%E6%89%93%E6%A0%87%E7%AD%BE)。使用轻量标签即可:
```bash
git tag r0.1.1 commit_id
```
然后使用`git push` 把本地 tag 传到远程 repo 上即可 
```bash
git push origin r0.1.1
```
打完 tag 后要记得编写 release note。

最后，发包准备工作的步骤可以总结为：  
- 用 develop 分支发 rc 包
- rc 包通过后拉分支
- 打 tag
- 发包
- 编写 release note



## ManyLinux

为了让有 C++ 依赖的 pip wheel 包可以适用于更多的 linux 系统，需要降低其本身的 glibc 的依赖。这就需要让 pip wheel 包在 manylinux 的 docker 下编包。关于查看系统的 glibc 版本，可以使用命令：`ldd --version`。

### Manylinux

关于 Manylinux，主要可以参考 Github 项目的说明[ github many linux](https://github.com/pypa/manylinux)。
manylinux1 支持 Centos5以上， manylinux2010 支持 Centos 6 以上，manylinux2014 支持Centos 7 以上。
目前使用 manylinux2010 基本可以满足所有的 linux 生产环境需求。（不建议使用manylinux1，系统较老，难度较大）

### 拉取 manylinux2010

```bash
docker pull quay.io/pypa/manylinux1_x86_64
```

### 使用 manylinux2010

启动 manylinux2010 docker。

```bash
docker run -it xxxxxx
```

在 manylinux2010 的docker环境自带 swig 和各种类型的 python 版本。这里注意不要自己下载 conda 来安装环境来编译 pip 包，要用 docker 本身的环境来编包。
设置python：

```bash
export PATH="/opt/python/cp37-cp37m/bin/:$PATH"
#export PATH="/opt/python/cp38-cp38/bin/:$PATH"
#export PATH="/opt/python/cp39-cp39/bin/:$PATH"
```

随后正常编包，编包后需要使用 [auditwheel](https://github.com/pypa/auditwheel) 来降低编好的wheel包的版本。
显示 wheel 包的 glibc 依赖版本

```bash
auditwheel show wheel包
```

降低 wheel包的版本

```bash
auditwheel repair wheel包
```
