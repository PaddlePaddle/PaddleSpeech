# 发包方法



## conda 代替系统依赖

conda可以用来代替一些 apt-get 安装的系统依赖，这样可以让项目适用于除了 ubuntu 以外的系统。

使用 conda 可以安装 sox, libsndfile，swig等 paddlespeech 需要的依赖：

```bash
conda install -y -c conda-forge sox libsndfile
```

部分系统会缺少libbzip2库，这个 paddlespeech 也是需要的，这也可以用 conda 安装：

```bash
conda install -y -c bzip2
```

conda也可以安装linux的C++的依赖：

```bash
conda install -y -c gcc_linux-64=8.4.0 gxx_linux-64=8.4.0
```

#### 剩余问题：使用conda环境编译kenlm失败。目前在conda环境下编译kenlm会出现链接失败的问题

目前知道需要的依赖：

```bash
conda install -c conda-forge eigen boost cmake
```



## python 编包方法

#### 创建 pypi的账号

创建 pypi 账号

#### 下载 twine

```
pip install twine
```

#### python 编包

编写好python包的setup.py, 然后使用如下命令编wheel包：

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

输入账号和密码后就可以上传wheel包了

#### 关于python 包的发包信息

主要可以参考这个[文档](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/?highlight=find_packages)



## Manylinux 降低含有 C++ 依赖的 pip 包的 glibc 依赖

为了让有C++依赖的 pip wheel 包可以适用于更多的 linux 系统，需要降低其本身的 glibc 的依赖。这就需要让 pip wheel 包在 manylinux 的 docker 下编包。关于查看系统的 glibc 版本，可以使用命令：`ldd --version`。

### Manylinux

关于Many Linux，主要可以参考 Github 项目的说明[ github many linux](https://github.com/pypa/manylinux)。
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

在 Many Linux 2010 的docker环境自带 swig 和各种类型的 python 版本。这里注意不要自己下载conda 来安装环境来编译 pip 包，要用 docker 本身的环境来编包。
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



## 区分 install 模式和 develop 模式

可以在setup.py 中划分 install 的依赖（基本依赖）和 develop 的依赖 （开发者额外依赖）。 setup_info 中 `install_requires` 设置 install 的依赖，而在 `extras_require` 中设置 `develop` key为 develop的依赖。
普通安装可以使用：

```bash
pip install . 
```

另外使用 pip 安装已发的包也是使用普通安装的：

```
pip install paddlespeech
```

而开发者可以使用如下方式安装，这样不仅会安装install的依赖，也会安装develop的依赖， 即：最后安装的依赖=install依赖 + develop依赖：

```bash
pip install -e .[develop]
```



## python 包的动态安装

可以使用 pip包来实现动态安装：

```python
import pip
if int(pip.__version__.split('.')[0]) > 9:
        from pip._internal import main
    else:
        from pip import main
    main(['install', package_name])
```
