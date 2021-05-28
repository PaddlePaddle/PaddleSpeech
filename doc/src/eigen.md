# Eigen

http://eigen.tuxfamily.org/dox/index.html

## 简介

Eigen是C++中可以用来调用并进行矩阵计算的一个库，简单了说它就是一个c++版本的matlab包。

## 安装

下载eigen：http://eigen.tuxfamily.org/index.php?title=Main_Page#Download

Eigen只包含头文件，因此它不需要实现编译，只需要你include到你的项目，指定好Eigen的头文件路径，编译项目即可。而且跨平台，当然这是必须的。

**方案一**

下载后，解压得到文件夹中，Eigen子文件夹便是我们需要的全部；如果你想使用Eigen暂不支持的特性，可以使用unsupported子文件夹。可以把Eigen/unsupported复制到任何你需要的地方。

**方案二** 

安装改包，其实就是把Eigen/unsupported的内容复制到“/usr/local/include/eigen3”下。在解压的文件夹下，新建build_dir，执行。

```
  cd build_dir
  cmake ../
  make install
```

详见INSTALL文件即可。

## 模块和头文件

Eigen库被分为一个Core模块和其他一些模块，每个模块有一些相应的头文件。 为了便于引用，Dense模块整合了一系列模块；Eigen模块整合了所有模块。一般情况下，`include<Eigen/Dense>` 就够了。

| Module      | Header file                 | Contents                                               |
| ----------- | --------------------------- | ------------------------------------------------------ |
| Core        | #include<Eigen/Core>        | Matrix和Array类，基础的线性代数运算和数组操作          |
| Geometry    | #include<Eigen/Geometry>    | 旋转、平移、缩放、2维和3维的各种变换                   |
| LU          | #include<Eigen/LU>          | 求逆，行列式，LU分解                                   |
| Cholesky    | #include <Eigen/Cholesky>   | LLT和LDLT Cholesky分解                                 |
| Householder | #include<Eigen/Householder> | 豪斯霍尔德变换，用于线性代数运算                       |
| SVD         | #include<Eigen/SVD>         | SVD分解                                                |
| QR          | #include<Eigen/QR>          | QR分解                                                 |
| Eigenvalues | #include<Eigen/Eigenvalues> | 特征值，特征向量分解                                   |
| Sparse      | #include<Eigen/Sparse>      | 稀疏矩阵的存储和一些基本的线性运算                     |
| 稠密矩阵    | #include<Eigen/Dense>       | 包含了Core/Geometry/LU/Cholesky/SVD/QR/Eigenvalues模块 |
| 矩阵        | #include<Eigen/Eigen>       | 包括Dense和Sparse(整合库)                              |

## 一个简单的例子

```
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
```

编译并执行:`g++ main.cpp -I /usr/local/include/eigen3/ -o maincpp`

```
 3  -1
2.5 1.5
```

Eigen头文件定义了许多类型，所有的类型都在Eigen的命名空间内。MatrixXd代表的是任意大小（X*X）的矩阵，并且每个元素为double类型。

## 例2： 矩阵和向量

再看另一个例子

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
int main()
{
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}
```

输出为：

```
m =
  94 89.8 43.5
49.4  101 86.8
88.3 29.8 37.8
m * v =
404
512
261
```

程序中定义了一个任意大小的矩阵，并用3`*`3的随机阵初始化。`MatrixXd::Constant`创建一个3*3的常量矩阵。

VectorXd表示列向量，并用*逗号初始化语法*来初始化。

在看同样功能的代码

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
int main()
{
  Matrix3d m = Matrix3d::Random();
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1,2,3);
  
  cout << "m * v =" << endl << m * v << endl;
}
```

MatrixXd表示是任意尺寸的矩阵，Matrix3d直接指定了3*3的大小。Vector3d也被直接初始化为[1,2,3]'的列向量。

使用固定大小的矩阵或向量有两个好处：编译更快；指定大小可以进行更为严格的检查。当然使用太多类别（Matrix3d、Matrix4d、Matrix5d...）会增加编译时间和可执行文件大小，原则建议使用4及以内的。

## Matrix类

在Eigen，所有的矩阵和向量都是**Matrix**模板类的对象，Vector只是一种特殊的矩阵（一行或者一列）。

Matrix有6个模板参数，主要使用前三个参数，剩下的有默认值。

```
Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
```

Scalar是表示元素的类型，RowsAtCompileTime为矩阵的行，ColsAtCompileTime为矩阵的列。

库中提供了一些类型便于使用，比如：

```
typedef Matrix<float, 4, 4> Matrix4f;
```

## Vectors向量

列向量

```
typedef Matrix<float, 3, 1> Vector3f;
```

行向量

```
typedef Matrix<int, 1, 2> RowVector2i;
```

## Dynamic

Eigen不只限于已知大小（编译阶段）的矩阵，有些矩阵的尺寸是运行时确定的，于是引入了一个特殊的标识符：Dynamic

```
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Matrix<int, Dynamic, 1> VectorXi;
Matrix<float, 3, Dynamic>
```

## 构造函数

默认的构造函数不执行任何空间分配，也不初始化矩阵的元素。

```
Matrix3f a;
MatrixXf b;
```

这里，a是一个3*3的矩阵，分配了float[9]的空间，但未初始化内部元素；b是一个动态大小的矩阵，定义是未分配空间(0*0)。

指定大小的矩阵，只是分配相应大小的空间，未初始化元素。

```
MatrixXf a(10,15);
VectorXf b(30);
```

这里，a是一个10*15的动态大小的矩阵，分配了空间但未初始化元素；b是一个30大小的向量，同样分配空间未初始化元素。

为了对固定大小和动态大小的矩阵提供统一的API，对指定大小的Matrix传递sizes也是合法的（传递也被忽略）。

```
Matrix3f a(3,3);
```

可以用构造函数提供4以内尺寸的vector的初始化。

```
Vector2d a(5.0, 6.0);
Vector3d b(5.0, 6.0, 7.0);
Vector4d c(5.0, 6.0, 7.0, 8.0);
```

## 获取元素

通过中括号获取元素，对于矩阵是：（行，列）；对于向量，只是传递它的索引，以0为起始。

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << "Here is the matrix m:\n" << m << std::endl;
  VectorXd v(2);
  v(0) = 4;
  v(1) = v(0) - 1;
  std::cout << "Here is the vector v:\n" << v << std::endl;
}
```

输出

```
Here is the matrix m:
  3  -1
2.5 1.5
Here is the vector v:
4
3
```

m(index)也可以用于获取矩阵元素，但取决于matrix的存储顺序，默认是按列存储的，当然也可以改为按行。

[]操作符可以用于向量元素的获取，但是不能用于matrix，因为C++中[]不能传递超过一个参数。

## 逗号初始化

```
Matrix3f m;
m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
std::cout << m;
```

## resizing

matrix的大小可以通过rows()、cols()、size()获取，resize()可以重新调整动态matrix的大小。

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
int main()
{
  MatrixXd m(2,5);
  m.resize(4,3);
  std::cout << "The matrix m is of size "
            << m.rows() << "x" << m.cols() << std::endl;
  std::cout << "It has " << m.size() << " coefficients" << std::endl;
  VectorXd v(2);
  v.resize(5);
  std::cout << "The vector v is of size " << v.size() << std::endl;
  std::cout << "As a matrix, v is of size "
            << v.rows() << "x" << v.cols() << std::endl;
}
```

输出：

```
The matrix m is of size 4x3
It has 12 coefficients
The vector v is of size 5
As a matrix, v is of size 5x1
```

如果matrix的实际大小不改变，resize函数不做任何操作。resize操作会执行析构函数：元素的值会被改变，如果不想改变执行 conservativeResize()。

为了统一API，所有的操作可用于指定大小的matrix，当然，实际中它不会改变大小。尝试去改变一个固定大小的matrix到一个不同的值，会出发警告失败。只有如下是合法的。

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
int main()
{
  Matrix4d m;
  m.resize(4,4); // no operation
  std::cout << "The matrix m is of size "
            << m.rows() << "x" << m.cols() << std::endl;
}
```

## assignment 和 resizing

assignment（分配）是复制一个矩阵到另外一个，操作符=。Eigen会自动resize左变量大小等于右变量大小，比如：

```
MatrixXf a(2,2);
std::cout << "a is of size " << a.rows() << "x" << a.cols() << std::endl;
MatrixXf b(3,3);
a = b;
std::cout << "a is now of size " << a.rows() << "x" << a.cols() << std::endl;

a is of size 2x2
a is now of size 3x3
```

当然，如果左边量是固定大小的，上面的resizing是不允许的。

## 固定尺寸 vs 动态尺寸

实际中，应该使用固定尺寸还是动态尺寸，简单的答案是：小的尺寸用固定的，大的尺寸用动态的。使用固定尺寸可以避免动态内存的开辟，固定尺寸只是一个普通数组。

```
Matrix4f mymatrix;` 等价于 `float mymatrix[16];
MatrixXf mymatrix(rows,columns);` 等价于 `float *mymatrix = new float[rows*columns];
```

使用固定尺寸(<=4*4)需要编译前知道矩阵大小，而且对于足够大的尺寸，如大于32，固定尺寸的收益可以忽略不计，而且可能导致栈崩溃。而且基于环境，Eigen会对动态尺寸做优化（类似于std::vector）

## 其他模板参数

上面只讨论了前三个参数，完整的模板参数如下：

```
Matrix<typename Scalar,
       int RowsAtCompileTime,
       int ColsAtCompileTime,
       int Options = 0,
       int MaxRowsAtCompileTime = RowsAtCompileTime,
       int MaxColsAtCompileTime = ColsAtCompileTime>
```

Options是一个比特标志位，这里，我们只介绍一种RowMajor，它表明matrix使用按行存储，默认是按列存储。`Matrix<float, 3, 3, RowMajor>`

MaxRowsAtCompileTime和MaxColsAtCompileTime表示在编译阶段矩阵的上限。主要是避免动态内存分配，使用数组。

```
Matrix<float, Dynamic, Dynamic, 0, 3, 4>` 等价于 `float [12]
```

## 一些方便的定义

Eigen定义了一些类型

- MatrixNt = Matrix<type, N, N> 特殊地有 MatrxXi = Matrix<int, Dynamic, Dynamic>
- VectorNt = Matrix<type, N, 1> 比如 Vector2f = Matrix<float, 2, 1>
- RowVectorNt = Matrix<type, 1, N> 比如 RowVector3d = Matrix<double, 1, 3>

N可以是2,3,4或X(Dynamic)

t可以是i(int)、f(float)、d(double)、cf(complex)、cd(complex)等。

# 矩阵和向量的运算

提供一些概述和细节：关于矩阵、向量以及标量的运算。

## 介绍

Eigen提供了matrix/vector的运算操作，既包括重载了c++的算术运算符+/-/*，也引入了一些特殊的运算比如点乘dot、叉乘cross等。

对于Matrix类（matrix和vectors）这些操作只支持线性代数运算，比如：matrix1*matrix2表示矩阵的乘机，vetor+scalar是不允许的。如果你想执行非线性代数操作，请看下一篇（暂时放下）。

## 加减

左右两侧变量具有相同的尺寸（行和列），并且元素类型相同（Eigen不自动转化类型）操作包括：

- 二元运算 + 如a+b
- 二元运算 - 如a-b
- 一元运算 - 如-a
- 复合运算 += 如a+=b
- 复合运算 -= 如a-=b

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
int main()
{
  Matrix2d a;
  a << 1, 2,
       3, 4;
  MatrixXd b(2,2);
  b << 2, 3,
       1, 4;
  std::cout << "a + b =\n" << a + b << std::endl;
  std::cout << "a - b =\n" << a - b << std::endl;
  std::cout << "Doing a += b;" << std::endl;
  a += b;
  std::cout << "Now a =\n" << a << std::endl;
  Vector3d v(1,2,3);
  Vector3d w(1,0,0);
  std::cout << "-v + w - v =\n" << -v + w - v << std::endl;
}
```

输出：

```
a + b =
3 5
4 8
a - b =
-1 -1
 2  0
Doing a += b;
Now a =
3 5
4 8
-v + w - v =
-1
-4
-6
```

## 标量乘法和除法

乘/除标量是非常简单的，如下：

- 二元运算 * 如matrix*scalar
- 二元运算 * 如scalar*matrix
- 二元运算 / 如matrix/scalar
- 复合运算 *= 如matrix*=scalar
- 复合运算 /= 如matrix/=scalar

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
int main()
{
  Matrix2d a;
  a << 1, 2,
       3, 4;
  Vector3d v(1,2,3);
  std::cout << "a * 2.5 =\n" << a * 2.5 << std::endl;
  std::cout << "0.1 * v =\n" << 0.1 * v << std::endl;
  std::cout << "Doing v *= 2;" << std::endl;
  v *= 2;
  std::cout << "Now v =\n" << v << std::endl;
}
```

结果

```
a * 2.5 =
2.5   5
7.5  10
0.1 * v =
0.1
0.2
0.3
Doing v *= 2;
Now v =
2
4
6
```

## 表达式模板

这里简单介绍，在高级主题中会详细解释。在Eigen中，线性运算比如+不会对变量自身做任何操作，会返回一个“表达式对象”来描述被执行的计算。当整个表达式被评估完（一般是遇到=号），实际的操作才执行。

这样做主要是为了优化，比如

```
VectorXf a(50), b(50), c(50), d(50);
...
a = 3*b + 4*c + 5*d;
```

Eigen会编译这段代码最终遍历一次即可运算完成。

```
for(int i = 0; i < 50; ++i)
  a[i] = 3*b[i] + 4*c[i] + 5*d[i];
```

因此，我们不必要担心大的线性表达式的运算效率。

## 转置和共轭

![img](https://images2015.cnblogs.com/blog/532915/201701/532915-20170124234218331-958991885.png) 表示transpose转置

![img](https://images2015.cnblogs.com/blog/532915/201701/532915-20170124234219644-1435768692.png) 表示conjugate共轭

![img](https://images2015.cnblogs.com/blog/532915/201701/532915-20170124234220831-1674600657.png) 表示adjoint(共轭转置) 伴随矩阵

```
MatrixXcf a = MatrixXcf::Random(2,2);
cout << "Here is the matrix a\n" << a << endl;
cout << "Here is the matrix a^T\n" << a.transpose() << endl;
cout << "Here is the conjugate of a\n" << a.conjugate() << endl;
cout << "Here is the matrix a^*\n" << a.adjoint() << endl;
```

输出

```
Here is the matrix a
 (-0.211,0.68) (-0.605,0.823)
 (0.597,0.566)  (0.536,-0.33)
Here is the matrix a^T
 (-0.211,0.68)  (0.597,0.566)
(-0.605,0.823)  (0.536,-0.33)
Here is the conjugate of a
 (-0.211,-0.68) (-0.605,-0.823)
 (0.597,-0.566)    (0.536,0.33)
Here is the matrix a^*
 (-0.211,-0.68)  (0.597,-0.566)
(-0.605,-0.823)    (0.536,0.33)
```

对于实数矩阵，conjugate不执行任何操作，adjoint等价于transpose。

transpose和adjoint会简单的返回一个代理对象并不对本省做转置。如果执行 `b=a.transpose()` ，a不变，转置结果被赋值给b。如果执行 `a=a.transpose()` Eigen在转置结束之前结果会开始写入a，所以a的最终结果不一定等于a的转置。

```
Matrix2i a; a << 1, 2, 3, 4;
cout << "Here is the matrix a:\n" << a << endl;
a = a.transpose(); // !!! do NOT do this !!!
cout << "and the result of the aliasing effect:\n" << a << endl;

Here is the matrix a:
1 2
3 4
and the result of the aliasing effect:
1 2
2 4
```

这被称为“别名问题”。在debug模式，当assertions打开的情况加，这种常见陷阱可以被自动检测到。

对 `a=a.transpose()` 这种操作，可以执行in-palce转置。类似还有adjointInPlace。

```
MatrixXf a(2,3); a << 1, 2, 3, 4, 5, 6;
cout << "Here is the initial matrix a:\n" << a << endl;
a.transposeInPlace();
cout << "and after being transposed:\n" << a << endl;

Here is the initial matrix a:
1 2 3
4 5 6
and after being transposed:
1 4
2 5
3 6
```

## 矩阵-矩阵的乘法和矩阵-向量的乘法

向量也是一种矩阵，实质都是矩阵-矩阵的乘法。

- 二元运算 *如a*b
- 复合运算 *=如a*=b

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
int main()
{
  Matrix2d mat;
  mat << 1, 2,
         3, 4;
  Vector2d u(-1,1), v(2,0);
  std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
  std::cout << "Here is mat*u:\n" << mat*u << std::endl;
  std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
  std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
  std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  mat = mat*mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;
}
```

输出

```
Here is mat*mat:
 7 10
15 22
Here is mat*u:
1
1
Here is u^T*mat:
2 2
Here is u^T*v:
-2
Here is u*v^T:
-2 -0
 2  0
Let's multiply mat by itself
Now mat is mat:
 7 10
15 22
```

`m=m*m`并不会导致别名问题，Eigen在这里做了特殊处理，引入了临时变量。实质将编译为：

```
tmp = m*m
m = tmp
```

如果你确定矩阵乘法是安全的（并没有别名问题），你可以使用noalias()函数来避免临时变量 `c.noalias() += a*b` 。

## 点运算和叉运算

dot()执行点积，cross()执行叉积，点运算得到1*1的矩阵。当然，点运算也可以用u.adjoint()*v来代替。

```
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
int main()
{
  Vector3d v(1,2,3);
  Vector3d w(0,1,2);
  cout << "Dot product: " << v.dot(w) << endl;
  double dp = v.adjoint()*w; // automatic conversion of the inner product to a scalar
  cout << "Dot product via a matrix product: " << dp << endl;
  cout << "Cross product:\n" << v.cross(w) << endl;
}
```

输出

```
Dot product: 8
Dot product via a matrix product: 8
Cross product:
 1
-2
 1
```

注意：点积只对三维vector有效。对于复数，Eigen的点积是第一个变量共轭和第二个变量的线性积。

## 基础的归约操作

Eigen提供了而一些归约函数：sum()、prod()、maxCoeff()和minCoeff()，他们对所有元素进行操作。

```
#include <iostream>
#include <Eigen/Dense>
using namespace std;
int main()
{
  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  cout << "Here is mat.sum():       " << mat.sum()       << endl;
  cout << "Here is mat.prod():      " << mat.prod()      << endl;
  cout << "Here is mat.mean():      " << mat.mean()      << endl;
  cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << endl;
  cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << endl;
  cout << "Here is mat.trace():     " << mat.trace()     << endl;
}
```

输出

```
Here is mat.sum():       10
Here is mat.prod():      24
Here is mat.mean():      2.5
Here is mat.minCoeff():  1
Here is mat.maxCoeff():  4
Here is mat.trace():     5
```

trace表示矩阵的迹，对角元素的和等价于 `a.diagonal().sum()` 。

minCoeff和maxCoeff函数也可以返回结果元素的位置信息。

```
Matrix3f m = Matrix3f::Random();
  std::ptrdiff_t i, j;
  float minOfM = m.minCoeff(&i,&j);
  cout << "Here is the matrix m:\n" << m << endl;
  cout << "Its minimum coefficient (" << minOfM 
       << ") is at position (" << i << "," << j << ")\n\n";
  RowVector4i v = RowVector4i::Random();
  int maxOfV = v.maxCoeff(&i);
  cout << "Here is the vector v: " << v << endl;
  cout << "Its maximum coefficient (" << maxOfV 
       << ") is at position " << i << endl;
```

输出

```
Here is the matrix m:
  0.68  0.597  -0.33
-0.211  0.823  0.536
 0.566 -0.605 -0.444
Its minimum coefficient (-0.605) is at position (2,1)

Here is the vector v:  1  0  3 -3
Its maximum coefficient (3) is at position 2
```

## 操作的有效性

Eigen会检测执行操作的有效性，在编译阶段Eigen会检测它们，错误信息是繁冗的，但错误信息会大写字母突出，比如:

```
Matrix3f m;
Vector4f v;
v = m*v;      // Compile-time error: YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES
```

当然动态尺寸的错误要在运行时发现，如果在debug模式，assertions会触发后，程序将崩溃。

```
MatrixXf m(3,3);
VectorXf v(4);
v = m * v; // Run-time assertion failure here: "invalid matrix product"
```

# Array类和元素级操作

## 为什么使用Array

相对于Matrix提供的线性代数运算，Array类提供了更为一般的数组功能。Array类为元素级的操作提供了有效途径，比如点加（每个元素加值）或两个数据相应元素的点乘。

## Array

Array是个类模板（类似于Matrx）,前三个参数是必须指定的，后三个是可选的，这点和Matrix是相同的。

```
Array<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
```

Eigen也提供的一些常用类定义，Array是同时支持一维和二维的（Matrix二维，Vector一维）。

| Type                          | Tyoedef  |
| ----------------------------- | -------- |
| Array<float,Dynamic,1>        | ArrayXf  |
| Array<float,3,1>              | Array3f  |
| Array<double,Dynamic,Dynamic> | ArrayXXd |
| Array<double,3,3>             | Array33d |

## 获取元素

读写操作重载于matrix， `<<` 可以用于初始化array或打印。

```
#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;
int main()
{
  ArrayXXf  m(2,2);
  
  // assign some values coefficient by coefficient
  m(0,0) = 1.0; m(0,1) = 2.0;
  m(1,0) = 3.0; m(1,1) = m(0,1) + m(1,0);
  
  // print values to standard output
  cout << m << endl << endl;
 
  // using the comma-initializer is also allowed
  m << 1.0,2.0,
       3.0,4.0;
     
  // print values to standard output
  cout << m << endl;
}
```

## 加法和减法

和matrix类似，要求array的尺寸一致。同时支持`array+/-scalar`的操作！

```
#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;
int main()
{
  ArrayXXf a(3,3);
  ArrayXXf b(3,3);
  a << 1,2,3,
       4,5,6,
       7,8,9;
  b << 1,2,3,
       1,2,3,
       1,2,3;
       
  // Adding two arrays
  cout << "a + b = " << endl << a + b << endl << endl;
  // Subtracting a scalar from an array
  cout << "a - 2 = " << endl << a - 2 << endl;
}
```

输出

```
a + b = 
 2  4  6
 5  7  9
 8 10 12

a - 2 = 
-1  0  1
 2  3  4
 5  6  7
```

## 乘法

支持array*scalar（类似于matrix），但是当执行array*array时，执行的是相应元素的乘积，因此两个array必须具有相同的尺寸。

```
int main()
{
  ArrayXXf a(2,2);
  ArrayXXf b(2,2);
  a << 1,2,
       3,4;
  b << 5,6,
       7,8;
  cout << "a * b = " << endl << a * b << endl;
}

a * b = 
 5 12
21 32
```

## 其他元素级操作

| Function | function                  |
| -------- | ------------------------- |
| abs      | 绝对值                    |
| sqrt     | 平方根                    |
| min(.)   | 两个array相应元素的最小值 |

```
int main()
{
  ArrayXf a = ArrayXf::Random(5);
  a *= 2;
  cout << "a =" << endl 
       << a << endl;
  cout << "a.abs() =" << endl 
       << a.abs() << endl;
  cout << "a.abs().sqrt() =" << endl 
       << a.abs().sqrt() << endl;
  cout << "a.min(a.abs().sqrt()) =" << endl 
       << a.min(a.abs().sqrt()) << endl;
}
```

## array和matrix之间的转换

当需要线性代数类操作时，请使用Matrix；但需要元素级操作时，需要使用Array。这样就需要提供两者的转化方法。

Matrix提供了.array()函数将它们转化为Array对象。

Array提供了.matrix()函数将它们转化为Matrix对象。

在Eigen，在表达式中混合Matrix和Array操作是被禁止的，但是可以将array表达式结果赋值为matrix。

另外，Matrix提供了cwiseProduct函数也实现了点乘。

```
#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;
int main()
{
  MatrixXf m(2,2);
  MatrixXf n(2,2);
  MatrixXf result(2,2);
  m << 1,2,
       3,4;
  n << 5,6,
       7,8;
  result = m * n;
  cout << "-- Matrix m*n: --" << endl << result << endl << endl;
  result = m.array() * n.array();
  cout << "-- Array m*n: --" << endl << result << endl << endl;
  result = m.cwiseProduct(n);
  cout << "-- With cwiseProduct: --" << endl << result << endl << endl;
  result = m.array() + 4;
  cout << "-- Array m + 4: --" << endl << result << endl << endl;
}
```

输出

```
-- Matrix m*n: --
19 22
43 50

-- Array m*n: --
 5 12
21 32

-- With cwiseProduct: --
 5 12
21 32

-- Array m + 4: --
5 6
7 8
```

类似， `array1.matrix() * array2.matrix()` 将执行矩阵乘法。

# 块操作

块是matrix或array中的矩形子部分。

## 使用块

函数.block()，有两种形式

| operation            | 构建一个动态尺寸的block | 构建一个固定尺寸的block |
| -------------------- | ----------------------- | ----------------------- |
| 起点(i,j)块大小(p,q) | .block(i,j,p,q)         | .block< p,q >(i,j)      |

Eigen中，索引从0开始。

两个版本都可以用于固定尺寸和动态尺寸的matrix/array。功能是等价的，只是固定尺寸的版本在block较小时速度更快一些。

```
int main()
{
  Eigen::MatrixXf m(4,4);
  m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
  cout << "Block in the middle" << endl;
  cout << m.block<2,2>(1,1) << endl << endl;
  for (int i = 1; i <= 3; ++i)
  {
    cout << "Block of size " << i << "x" << i << endl;
    cout << m.block(0,0,i,i) << endl << endl;
  }
}
```

输出

```
Block in the middle
 6  7
10 11

Block of size 1x1
1

Block of size 2x2
1 2
5 6

Block of size 3x3
 1  2  3
 5  6  7
 9 10 11
```

作为左值

```
int main()
{
  Array22f m;
  m << 1,2,
       3,4;
  Array44f a = Array44f::Constant(0.6);
  cout << "Here is the array a:" << endl << a << endl << endl;
  a.block<2,2>(1,1) = m;
  cout << "Here is now a with m copied into its central 2x2 block:" << endl << a << endl << endl;
  a.block(0,0,2,3) = a.block(2,1,2,3);
  cout << "Here is now a with bottom-right 2x3 block copied into top-left 2x2 block:" << endl << a << endl << endl;
}
```

输出

```
Here is the array a:
0.6 0.6 0.6 0.6
0.6 0.6 0.6 0.6
0.6 0.6 0.6 0.6
0.6 0.6 0.6 0.6

Here is now a with m copied into its central 2x2 block:
0.6 0.6 0.6 0.6
0.6   1   2 0.6
0.6   3   4 0.6
0.6 0.6 0.6 0.6

Here is now a with bottom-right 2x3 block copied into top-left 2x2 block:
  3   4 0.6 0.6
0.6 0.6 0.6 0.6
0.6   3   4 0.6
0.6 0.6 0.6 0.6
```

## 行和列

| Operation | Method        |
| --------- | ------------- |
| ith row   | matrix.row(i) |
| jth colum | matrix.col(j) |

```
int main()
{
  Eigen::MatrixXf m(3,3);
  m << 1,2,3,
       4,5,6,
       7,8,9;
  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "2nd Row: " << m.row(1) << endl;
  m.col(2) += 3 * m.col(0);
  cout << "After adding 3 times the first column into the third column, the matrix m is:\n";
  cout << m << endl;
}
```

输出

```
Here is the matrix m:
1 2 3
4 5 6
7 8 9
2nd Row: 4 5 6
After adding 3 times the first column into the third column, the matrix m is:
 1  2  6
 4  5 18
 7  8 30
```

## 角相关操作

| operation  | dynamic-size block             | fixed-size block                   |
| ---------- | ------------------------------ | ---------------------------------- |
| 左上角p\*q | matrix.topLeftCorner(p,q);     | matrix.topLeftCorner< p,q >();     |
| 左下角p\*q | matrix.bottomLeftCorner(p,q);  | matrix.bottomLeftCorner< p,q >();  |
| 右上角p\*q | matrix.topRightCorner(p,q);    | matrix.topRightCorner< p,q >();    |
| 右下角p\*q | matrix.bottomRightCorner(p,q); | matrix.bottomRightCorner< p,q >(); |
| 前q行      | matrix.topRows(q);             | matrix.topRows< q >();             |
| 后q行      | matrix.bottomRows(q);          | matrix.bottomRows< q >();          |
| 左p列      | matrix.leftCols(p);            | matrix.leftCols< p >();            |
| 右p列      | matrix.rightCols(p);           | matrix.rightCols< p >();           |

```
int main()
{
  Eigen::Matrix4f m;
  m << 1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10,11,12,
       13,14,15,16;
  cout << "m.leftCols(2) =" << endl << m.leftCols(2) << endl << endl;
  cout << "m.bottomRows<2>() =" << endl << m.bottomRows<2>() << endl << endl;
  m.topLeftCorner(1,3) = m.bottomRightCorner(3,1).transpose();
  cout << "After assignment, m = " << endl << m << endl;
}
```

输出

```
m.leftCols(2) =
 1  2
 5  6
 9 10
13 14

m.bottomRows<2>() =
 9 10 11 12
13 14 15 16

After assignment, m = 
 8 12 16  4
 5  6  7  8
 9 10 11 12
13 14 15 16
```

## vectors的块操作

| operation      | dynamic-size block   | fixed-size block        |
| -------------- | -------------------- | ----------------------- |
| 前n个          | vector.head(n);      | vector.head< n >();     |
| 后n个          | vector.tail(n);      | vector.tail< n >();     |
| i起始的n个元素 | vector.segment(i,n); | vector.segment< n >(i); |

# 高级初始化方法

本篇介绍几种高级的矩阵初始化方法，重点介绍逗号初始化和特殊矩阵（单位阵、零阵）。

## 逗号初始化

Eigen提供了逗号操作符允许我们方便地为矩阵/向量/数组中的元素赋值。顺序是从左上到右下：自左到右，从上至下。对象的尺寸需要事先指定，初始化的参数也应该和要操作的元素数目一致。

```
Matrix3f m;
m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
std::cout << m;
```

初始化列表不仅可以是数值也可以是vectors或matrix。

```
RowVectorXd vec1(3);
vec1 << 1, 2, 3;
std::cout << "vec1 = " << vec1 << std::endl;
RowVectorXd vec2(4);
vec2 << 1, 4, 9, 16;
std::cout << "vec2 = " << vec2 << std::endl;
RowVectorXd joined(7);
joined << vec1, vec2;
std::cout << "joined = " << joined << std::endl;
```

输出

```
vec1 = 1 2 3
vec2 =  1  4  9 16
joined =  1  2  3  1  4  9 16
```

也可以使用块结构。

```
MatrixXf matA(2, 2);
matA << 1, 2, 3, 4;
MatrixXf matB(4, 4);
matB << matA, matA/10, matA/10, matA;
std::cout << matB << std::endl;
```

输出

```
  1   2 0.1 0.2
  3   4 0.3 0.4
0.1 0.2   1   2
0.3 0.4   3   4
```

同时逗号初始化方式也可以用来为块表达式赋值。

```
Matrix3f m;
m.row(0) << 1, 2, 3;
m.block(1,0,2,2) << 4, 5, 7, 8;
m.col(2).tail(2) << 6, 9;                   
std::cout << m;

1 2 3
4 5 6
7 8 9
```

## 特殊的矩阵和向量

零阵：类的静态成员函数Zero()，有三种定义形式。

```
std::cout << "A fixed-size array:\n";
Array33f a1 = Array33f::Zero();
std::cout << a1 << "\n\n";
std::cout << "A one-dimensional dynamic-size array:\n";
ArrayXf a2 = ArrayXf::Zero(3);
std::cout << a2 << "\n\n";
std::cout << "A two-dimensional dynamic-size array:\n";
ArrayXXf a3 = ArrayXXf::Zero(3, 4);
std::cout << a3 << "\n";
```

输出

```
A fixed-size array:
0 0 0
0 0 0
0 0 0

A one-dimensional dynamic-size array:
0
0
0

A two-dimensional dynamic-size array:
0 0 0 0
0 0 0 0
0 0 0 0
```

类似地，还有常量矩阵：Constant([rows],[cols],value)，Random()随机矩阵。

单位阵Identity()方法只能使用与Matrix不使用Array，因为单位阵是个线性代数概念。

LinSpaced(size, low, high)可以从low到high等间距的size长度的序列，适用于vector和一维数组。

```
ArrayXXf table(10, 4);
table.col(0) = ArrayXf::LinSpaced(10, 0, 90);
table.col(1) = M_PI / 180 * table.col(0);
table.col(2) = table.col(1).sin();
table.col(3) = table.col(1).cos();
std::cout << "  Degrees   Radians      Sine    Cosine\n";
std::cout << table << std::endl;
```

输出

```
  Degrees   Radians      Sine    Cosine
        0         0         0         1
       10     0.175     0.174     0.985
       20     0.349     0.342      0.94
       30     0.524       0.5     0.866
       40     0.698     0.643     0.766
       50     0.873     0.766     0.643
       60      1.05     0.866       0.5
       70      1.22      0.94     0.342
       80       1.4     0.985     0.174
       90      1.57         1 -4.37e-08
```

### 功能函数

Eigen也提供可同样功能的函数：setZero(), MatrixBase::setIdentity()和 DenseBase::setLinSpaced()。

```
const int size = 6;
MatrixXd mat1(size, size);
mat1.topLeftCorner(size/2, size/2)     = MatrixXd::Zero(size/2, size/2);
mat1.topRightCorner(size/2, size/2)    = MatrixXd::Identity(size/2, size/2);
mat1.bottomLeftCorner(size/2, size/2)  = MatrixXd::Identity(size/2, size/2);
mat1.bottomRightCorner(size/2, size/2) = MatrixXd::Zero(size/2, size/2);
std::cout << mat1 << std::endl << std::endl;
MatrixXd mat2(size, size);
mat2.topLeftCorner(size/2, size/2).setZero();
mat2.topRightCorner(size/2, size/2).setIdentity();
mat2.bottomLeftCorner(size/2, size/2).setIdentity();
mat2.bottomRightCorner(size/2, size/2).setZero();
std::cout << mat2 << std::endl << std::endl;
MatrixXd mat3(size, size);
mat3 << MatrixXd::Zero(size/2, size/2), MatrixXd::Identity(size/2, size/2),
        MatrixXd::Identity(size/2, size/2), MatrixXd::Zero(size/2, size/2);
std::cout << mat3 << std::endl;
```

输出均为

```
0 0 0 1 0 0
0 0 0 0 1 0
0 0 0 0 0 1
1 0 0 0 0 0
0 1 0 0 0 0
0 0 1 0 0 0
```

三种赋值（初始化）的方式逗号初始化、特殊阵的静态方法和功能函数setXxx()。

## 表达式变量

上面的静态方法如 Zero()、Constant()并不是直接返回一个矩阵或数组，实际上它们返回的是是‘expression object’，只是临时被使用/被用于优化。

```
m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
```

`MatrixXf::Constant(3,3,1.2)`构建的是一个3*3的矩阵表达式（临时变量）。

逗号初始化的方式也可以构建这种临时变量，这是为了获取真正的矩阵需要调用finished()函数：

```
MatrixXf mat = MatrixXf::Random(2, 3);
std::cout << mat << std::endl << std::endl;
mat = (MatrixXf(2,2) << 0, 1, 1, 0).finished() * mat;
std::cout << mat << std::endl;
```

输出

```
  0.68  0.566  0.823
-0.211  0.597 -0.605

-0.211  0.597 -0.605
  0.68  0.566  0.823
```

# 归约、迭代器和广播

## 归约

在Eigen中，有些函数可以统计matrix/array的某类特征，返回一个标量。

```
int main()
{
  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  cout << "Here is mat.sum():       " << mat.sum()       << endl;
  cout << "Here is mat.prod():      " << mat.prod()      << endl;
  cout << "Here is mat.mean():      " << mat.mean()      << endl;
  cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << endl;
  cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << endl;
  cout << "Here is mat.trace():     " << mat.trace()     << endl;
}
```

### 范数计算

L2范数 squareNorm()，等价于计算vector的自身点积，norm()返回squareNorm的开方根。

这些操作应用于matrix，norm() 会返回Frobenius或Hilbert-Schmidt范数。

如果你想使用其他Lp范数，可以使用lpNorm< p >()方法。p可以取Infinity，表示L∞范数。

```
int main()
{
  VectorXf v(2);
  MatrixXf m(2,2), n(2,2);
  
  v << -1,
       2;
  
  m << 1,-2,
       -3,4;
  cout << "v.squaredNorm() = " << v.squaredNorm() << endl;
  cout << "v.norm() = " << v.norm() << endl;
  cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << endl;
  cout << "v.lpNorm<Infinity>() = " << v.lpNorm<Infinity>() << endl;
  cout << endl;
  cout << "m.squaredNorm() = " << m.squaredNorm() << endl;
  cout << "m.norm() = " << m.norm() << endl;
  cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << endl;
  cout << "m.lpNorm<Infinity>() = " << m.lpNorm<Infinity>() << endl;
}
```

输出

```
v.squaredNorm() = 5
v.norm() = 2.23607
v.lpNorm<1>() = 3
v.lpNorm<Infinity>() = 2

m.squaredNorm() = 30
m.norm() = 5.47723
m.lpNorm<1>() = 10
m.lpNorm<Infinity>() = 4
```

**Operator norm**: 1-norm和∞-norm可以通过其他方式得到。

```
int main()
{
  MatrixXf m(2,2);
  m << 1,-2,
       -3,4;
  cout << "1-norm(m)     = " << m.cwiseAbs().colwise().sum().maxCoeff()
       << " == "             << m.colwise().lpNorm<1>().maxCoeff() << endl;
  cout << "infty-norm(m) = " << m.cwiseAbs().rowwise().sum().maxCoeff()
       << " == "             << m.rowwise().lpNorm<1>().maxCoeff() << endl;
}

1-norm(m)     = 6 == 6
infty-norm(m) = 7 == 7
```

### 布尔归约

all()=true matrix/array中的所有算术是true any()=true matrix/array中至少有一个元素是true count() 返回为true元素的数目

```
#include <Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;
int main()
{
  ArrayXXf a(2,2);
  
  a << 1,2,
       3,4;
  cout << "(a > 0).all()   = " << (a > 0).all() << endl;
  cout << "(a > 0).any()   = " << (a > 0).any() << endl;
  cout << "(a > 0).count() = " << (a > 0).count() << endl;
  cout << endl;
  cout << "(a > 2).all()   = " << (a > 2).all() << endl;
  cout << "(a > 2).any()   = " << (a > 2).any() << endl;
  cout << "(a > 2).count() = " << (a > 2).count() << endl;
}
```

输出

```
(a > 0).all()   = 1
(a > 0).any()   = 1
(a > 0).count() = 4

(a > 2).all()   = 0
(a > 2).any()   = 1
(a > 2).count() = 2
```

## 迭代器(遍历)

当我们想获取某元素在Matrix或Array中的位置的时候，迭代器是必须的。常用的有：minCoeff和maxCoeff。

```
int main()
{
  Eigen::MatrixXf m(2,2);
  
  m << 1, 2,
       3, 4;
  //get location of maximum
  MatrixXf::Index maxRow, maxCol;
  float max = m.maxCoeff(&maxRow, &maxCol);
  //get location of minimum
  MatrixXf::Index minRow, minCol;
  float min = m.minCoeff(&minRow, &minCol);
  cout << "Max: " << max <<  ", at: " <<
     maxRow << "," << maxCol << endl;
  cout << "Min: " << min << ", at: " <<
     minRow << "," << minCol << endl;
}

Max: 4, at: 1,1
Min: 1, at: 0,0
```

## 部分归约

Eigen中支持对Matrx或Array的行/行进行归约操作。部分归约可以使用colwise()/rowwise()函数。

```
int main()
{
  Eigen::MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  
  std::cout << "Column's maximum: " << std::endl
   << mat.colwise().maxCoeff() << std::endl;
}

Column's maximum: 
3 2 7 9
```

类似，针对行也可以，只是返回的是列向量而已。

```
int main()
{
  Eigen::MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  
  std::cout << "Row's maximum: " << std::endl
   << mat.rowwise().maxCoeff() << std::endl;
}

Row's maximum: 
9
7
```

## 结合部分归约和其他操作

例子：寻找和最大的列向量。

```
int main()
{
  MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  
  MatrixXf::Index   maxIndex;
  float maxNorm = mat.colwise().sum().maxCoeff(&maxIndex);
  
  std::cout << "Maximum sum at position " << maxIndex << std::endl;
  std::cout << "The corresponding vector is: " << std::endl;
  std::cout << mat.col( maxIndex ) << std::endl;
  std::cout << "And its sum is is: " << maxNorm << std::endl;
}
```

输出

```
Maximum sum at position 2
The corresponding vector is: 
6
7
And its sum is is: 13
```

## 广播

广播是针对vector的，将vector沿行/列重复构建一个matrix，便于后期运算。

```
int main()
{
  Eigen::MatrixXf mat(2,4);
  Eigen::VectorXf v(2);
  
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
         
  v << 0,
       1;
       
  //add v to each column of m
  mat.colwise() += v;
  
  std::cout << "Broadcasting result: " << std::endl;
  std::cout << mat << std::endl;
}
```

输出

```
Broadcasting result: 
1 2 6 9
4 2 8 3
```

注意：对Array类型，*=，/=和/这些操作可以进行行/列级的操作，但不使用与Matrix，因为会与矩阵乘混淆。

## 结合广播和其他操作

示例：计算矩阵中哪列与目标向量距离最近。

```
int main()
{
  Eigen::MatrixXf m(2,4);
  Eigen::VectorXf v(2);
  
  m << 1, 23, 6, 9,
       3, 11, 7, 2;
       
  v << 2,
       3;
  MatrixXf::Index index;
  // find nearest neighbour
  (m.colwise() - v).colwise().squaredNorm().minCoeff(&index);
  cout << "Nearest neighbour is column " << index << ":" << endl;
  cout << m.col(index) << endl;
}
```

输出

```
Nearest neighbour is column 0:
1
3
```

# 原生缓存的接口：Map类

这篇将解释Eigen如何与原生raw C/C++ 数组混合编程。

## 简介

Eigen中定义了一系列的vector和matrix，相比copy数据，更一般的方式是复用数据的内存，将它们转变为Eigen类型。Map类很好地实现了这个功能。

## Map类型

Map的定义

```
Map<Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime> >
```

默认情况下，Mao只需要一个模板参数。

为了构建Map变量，我们需要其余的两个信息：一个指向元素数组的指针，Matrix/vector的尺寸。定义一个float类型的矩阵： `Map<MatrixXf> mf(pf,rows,columns);` pf是一个数组指针float *。

固定尺寸的整形vector声明： `Map<const Vector4i> mi(pi);`

注意:Map没有默认的构造函数，你需要传递一个指针来初始化对象。

Mat是灵活地足够去容纳多种不同的数据表示，其他的两个模板参数：

```
Map<typename MatrixType,
    int MapOptions,
    typename StrideType>
```

MapOptions标识指针是否是对齐的（Aligned），默认是Unaligned。

StrideType表示内存数组的组织方式：行列的步长。

```
int array[8];
for(int i = 0; i < 8; ++i) array[i] = i;
cout << "Column-major:\n" << Map<Matrix<int,2,4> >(array) << endl;
cout << "Row-major:\n" << Map<Matrix<int,2,4,RowMajor> >(array) << endl;
cout << "Row-major using stride:\n" <<
  Map<Matrix<int,2,4>, Unaligned, Stride<1,4> >(array) << endl;
```

输出

```
Column-major:
0 2 4 6
1 3 5 7
Row-major:
0 1 2 3
4 5 6 7
Row-major using stride:
0 1 2 3
4 5 6 7
```

## 使用Map变量

可以像Eigen的其他类型一样来使用Map类型。

```
typedef Matrix<float,1,Dynamic> MatrixType;
typedef Map<MatrixType> MapType;
typedef Map<const MatrixType> MapTypeConst;   // a read-only map
const int n_dims = 5;
  
MatrixType m1(n_dims), m2(n_dims);
m1.setRandom();
m2.setRandom();
float *p = &m2(0);  // get the address storing the data for m2
MapType m2map(p,m2.size());   // m2map shares data with m2
MapTypeConst m2mapconst(p,m2.size());  // a read-only accessor for m2
cout << "m1: " << m1 << endl;
cout << "m2: " << m2 << endl;
cout << "Squared euclidean distance: " << (m1-m2).squaredNorm() << endl;
cout << "Squared euclidean distance, using map: " <<
  (m1-m2map).squaredNorm() << endl;
m2map(3) = 7;   // this will change m2, since they share the same array
cout << "Updated m2: " << m2 << endl;
cout << "m2 coefficient 2, constant accessor: " << m2mapconst(2) << endl;
/* m2mapconst(2) = 5; */   // this yields a compile-time error
```

输出

```
m1:   0.68 -0.211  0.566  0.597  0.823
m2: -0.605  -0.33  0.536 -0.444  0.108
Squared euclidean distance: 3.26
Squared euclidean distance, using map: 3.26
Updated m2: -0.605  -0.33  0.536      7  0.108
m2 coefficient 2, constant accessor: 0.536
```

Eigen提供的函数都兼容Map对象。

## 改变mapped数组

Map对象声明后，可以通过C++的placement new语法来改变Map的数组。

```
int data[] = {1,2,3,4,5,6,7,8,9};
Map<RowVectorXi> v(data,4);
cout << "The mapped vector v is: " << v << "\n";
new (&v) Map<RowVectorXi>(data+4,5);
cout << "Now v is: " << v << "\n";

The mapped vector v is: 1 2 3 4
Now v is: 5 6 7 8 9
```

Eigen并没有为matrix提供直接的Reshape和Slicing的API，但是这些特性可以通过Map类来实现。

## Reshape

reshape操作是改变matrix的尺寸大小但保持元素不变。采用的方法是创建一个不同“视图” Map。

```
MatrixXf M1(3,3);    // Column-major storage
M1 << 1, 2, 3,
      4, 5, 6,
      7, 8, 9;
Map<RowVectorXf> v1(M1.data(), M1.size());
cout << "v1:" << endl << v1 << endl;
Matrix<float,Dynamic,Dynamic,RowMajor> M2(M1);
Map<RowVectorXf> v2(M2.data(), M2.size());
cout << "v2:" << endl << v2 << endl;
```

输出

```
v1:
1 4 7 2 5 8 3 6 9
v2:
1 2 3 4 5 6 7 8 9
```

reshape 2*6的矩阵到 6*2

```
MatrixXf M1(2,6);    // Column-major storage
M1 << 1, 2, 3,  4,  5,  6,
      7, 8, 9, 10, 11, 12;
Map<MatrixXf> M2(M1.data(), 6,2);
cout << "M2:" << endl << M2 << endl;
```

输出

```
M2:
 1  4
 7 10
 2  5
 8 11
 3  6
 9 12
```

## Slicing

也是通过Map实现的，比如：每p个元素获取一个。

```
RowVectorXf v = RowVectorXf::LinSpaced(20,0,19);
cout << "Input:" << endl << v << endl;
Map<RowVectorXf,0,InnerStride<2> > v2(v.data(), v.size()/2);
cout << "Even:" << v2 << endl;
```

输出

```
Input:
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
Even: 0  2  4  6  8 10 12 14 16 18
```

## 混淆

在Eigen中，当变量同时出现在左值和右值，赋值操作可能会带来混淆问题。这一篇将解释什么是混淆，什么时候是有害的，怎么使用做。

## 例子

```
MatrixXi mat(3,3); 
mat << 1, 2, 3,   4, 5, 6,   7, 8, 9;
cout << "Here is the matrix mat:\n" << mat << endl;
// This assignment shows the aliasing problem
mat.bottomRightCorner(2,2) = mat.topLeftCorner(2,2);
cout << "After the assignment, mat = \n" << mat << endl;
```

输出

```
Here is the matrix mat:
1 2 3
4 5 6
7 8 9
After the assignment, mat = 
1 2 3
4 1 2
7 4 1
```

在 `mat.bottomRightCorner(2,2) = mat.topLeftCorner(2,2);` 赋值中展示了混淆。

mat(1,1) 在bottomRightCorner(2,2)和topLeftCorner(2,2)都存在。赋值结果中mat(2,2)本应该赋予操作前mat(1,1)的值=5。但是，最终程序结果mat(2,2)=1。原因是Eigen使用了lazy evaluation（懒惰评估），上面等价于

```
mat(1,1) = mat(0,0);
mat(1,2) = mat(0,1);
mat(2,1) = mat(1,0);
mat(2,2) = mat(1,1);
```

下面会解释如何通过eval()来解决这个问题。

混淆还会在缩小矩阵时出现，比如 `vec = vec.head(n)` 和 `mat = mat.block(i,j,r,c)`。

一般来说，混淆在编译阶段很难被检测到。比如第一个例子，如果mat再大一些可能就不会出现混淆了。但是Eigen可以在运行时检测某些混淆，如前面讲的例子。

```
Matrix2i a; a << 1, 2, 3, 4;
cout << "Here is the matrix a:\n" << a << endl;
a = a.transpose(); // !!! do NOT do this !!!
cout << "and the result of the aliasing effect:\n" << a << endl;
Here is the matrix a:
1 2
3 4
and the result of the aliasing effect:
1 2
2 4
```

我们可以通过EIGEN_NO_DEBUG宏，在编译时关闭运行时的断言。

## 解决混淆问题

Eigen需要把右值赋值为一个临时matrix/array，然后再将临时值赋值给左值，便可以解决混淆。eval()函数实现了这个功能。

```
MatrixXi mat(3,3); 
mat << 1, 2, 3,   4, 5, 6,   7, 8, 9;
cout << "Here is the matrix mat:\n" << mat << endl;
// The eval() solves the aliasing problem
mat.bottomRightCorner(2,2) = mat.topLeftCorner(2,2).eval();
cout << "After the assignment, mat = \n" << mat << endl;
```

输出

```
Here is the matrix mat:
1 2 3
4 5 6
7 8 9
After the assignment, mat = 
1 2 3
4 1 2
7 4 5
```

同样： `a = a.transpose().eval();` ，当然我们最好使用 transposeInPlace()。如果存在xxxInPlace函数，推荐使用这类函数，它们更加清晰地标明了你在做什么。提供的这类函数：

| Origin                  | In-place                       |
| ----------------------- | ------------------------------ |
| MatrixBase::adjoint()   | MatrixBase::adjointInPlace()   |
| DenseBase::reverse()    | DenseBase::reverseInPlace()    |
| LDLT::solve()           | LDLT::solveInPlace()           |
| LLT::solve()            | LLT::solveInPlace()            |
| TriangularView::solve() | TriangularView::solveInPlace() |
| DenseBase::transpose()  | DenseBase::transposeInPlace()  |

而针对`vec = vec.head(n)`这种情况，推荐使用`conservativeResize()`。

## 混淆和component级的操作。

组件级是指整体的操作，比如matrix加法、scalar乘、array乘等，这类操作是安全的，不会出现混淆。

```
MatrixXf mat(2,2); 
mat << 1, 2,  4, 7;
cout << "Here is the matrix mat:\n" << mat << endl << endl;
mat = 2 * mat;
cout << "After 'mat = 2 * mat', mat = \n" << mat << endl << endl;
mat = mat - MatrixXf::Identity(2,2);
cout << "After the subtraction, it becomes\n" << mat << endl << endl;
ArrayXXf arr = mat;
arr = arr.square();
cout << "After squaring, it becomes\n" << arr << endl << endl;
```

输出

```
Here is the matrix mat:
1 2
4 7

After 'mat = 2 * mat', mat = 
 2  4
 8 14

After the subtraction, it becomes
 1  4
 8 13

After squaring, it becomes
  1  16
 64 169
```

## 混淆和矩阵的乘法

在Eigen中，矩阵的乘法一般都会出现混淆。除非是方阵（实质是元素级的乘）。

```
MatrixXf matA(2,2); 
matA << 2, 0,  0, 2;
matA = matA * matA;
cout << matA;

4 0
0 4
```

其他的操作，Eigen默认都是存在混淆的。所以Eigen对矩阵乘法自动引入了临时变量，对的`matA=matA*matA`这是必须的，但是对`matB=matA*matA`这样便是不必要的了。我们可以使用noalias()函数来声明这里没有混淆，matA*matA的结果可以直接赋值为matB。

```
matB.noalias() = matA * matA;
```

从Eigen3.3开始，如果目标矩阵resize且结果不直接赋值给目标矩阵，默认不存在混淆。

```
MatrixXf A(2,2), B(3,2);
B << 2, 0,  0, 3, 1, 1;
A << 2, 0, 0, -2;
A = (B * A).cwiseAbs();//cwiseAbs（）不直接赋给目标
//A = (B * A).eval().cwiseAbs()
cout << A;
```

当然，对于任何混淆问题，都可以通过`matA=(matB*matA).eval()` 来解决。

## 总结

当相同的矩阵或array在等式左右都出现时，很容易出现混淆。

1. compnent级别的操作不用考虑混淆。
2. 矩阵相乘，Eigen默认会解决混淆问题，如果你确定不会出现混淆，可以使用noalias（）来提效。
3. 混淆出现时，可以用eval()和xxxInPlace()函数解决。

## 存储顺序

对于矩阵和二维数组有两种存储方式，列优先和行优先。

假设矩阵：

![img](https://images2015.cnblogs.com/blog/532915/201701/532915-20170125204938597-130252060.png)

按行优先存储，内存中形式如下：

```
8 2 2 9 9 1 4 4 3 5 4 5
```

列优先，内存格式：

```
8 9 3 2 1 5 2 4 4 9 4 5
Matrix<int, 3, 4, ColMajor> Acolmajor;
Acolmajor << 8, 2, 2, 9,
             9, 1, 4, 4,
             3, 5, 4, 5;
cout << "The matrix A:" << endl;
cout << Acolmajor << endl << endl; 
cout << "In memory (column-major):" << endl;
for (int i = 0; i < Acolmajor.size(); i++)
  cout << *(Acolmajor.data() + i) << "  ";
cout << endl << endl;
Matrix<int, 3, 4, RowMajor> Arowmajor = Acolmajor;
cout << "In memory (row-major):" << endl;
for (int i = 0; i < Arowmajor.size(); i++)
  cout << *(Arowmajor.data() + i) << "  ";
cout << endl;
```

输出

```
The matrix A:
8 2 2 9
9 1 4 4
3 5 4 5

In memory (column-major):
8  9  3  2  1  5  2  4  4  9  4  5  

In memory (row-major):
8  2  2  9  9  1  4  4  3  5  4  5 
```

PlainObjectBase::data()函数可以返回矩阵中第一个元素的内存位置。

## 存储顺序及选择

Matrix类模板中可以设定存储的方向，RowMajor表示行优先，ColMajor表示列优先。默认是列优先。

如何选择存储方式呢？

1. 如果要和其他库合作开发，为了转化方便，可以选择同样的存储方式。
2. 应用中涉及大量行遍历操作，应该选择行优先，寻址更快。反之亦然。
3. 默认是列优先，而且大多库都是按照这个顺序的，默认的不失为较好的。

## 总结

本来想春节前任务比较少，翻译完所有的Eigen系列的。但是我的目的是为了使用google的非线性优化库[ceres](http://ceres-solver.org/installation.html#getting-the-source-code)，介绍了这些基本知识也够用了，如果遇到不清楚的函数可以直接到Eigen的官网查询。

## Reference

* https://www.cnblogs.com/houkai/p/6347408.html
* https://www.cnblogs.com/houkai/p/6347648.html
* https://www.cnblogs.com/houkai/p/6348044.html
* https://www.cnblogs.com/houkai/p/6349970.html
* https://www.cnblogs.com/houkai/p/6349974.html
* https://www.cnblogs.com/houkai/p/6351358.html
* https://www.cnblogs.com/houkai/p/6351609.html
* https://www.cnblogs.com/houkai/p/6349981.html
* https://www.cnblogs.com/houkai/p/6349988.html
* https://www.cnblogs.com/houkai/p/6349990.html
* https://www.cnblogs.com/houkai/p/6349991.html
* [**Eigen Cheat sheet**](https://gist.github.com/gocarlos/c91237b02c120c6319612e42fa196d77#file-eigen-cheat-sheet)

