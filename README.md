# 图像读取+MINST读取 部分方法介绍

## ```FlatRoot::fileFR```
读取某```dirName```路径下所有满足```allowedExtensions```后缀名的文件 返回```ArrayList<File>```变量
### 函数原型
```{java}
public static ArrayList<File> fileFR(String dirName, String[] allowedExtensions)
```
### 例程
```{java}
String[] allowedExtensions = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};
List<File> files = FlatRoot.fileFR("F:/Programs/moduleimage/ImageDemo", allowedExtensions);
```
## ```ImageRead::files2INDArray```
将```List<File> files```中的文件作为图片读入```INDArray```中
### 函数原型
```{java}
public static INDArray files2INDArray(List<File> files, int channel, int row, int col, Function<BufferedImage, BufferedImage>... functions)
```
### 参数说明
#### ```functions```
该方法的读取过程是 先从文件中读图像数据到```BufferedImage```对象 然后再将读取到的一组```BufferedImage```对象转换成```INDArray```

在从文件中读图像数据到```BufferedImage```对象之后 将读取到的一组```BufferedImage```对象转换成```INDArray```之前
该方法将对每个```BufferedImage```对象按照数组顺序执行```functions```中储存的操作
#### ```row``` / ```col```
将该组图片以宽×高为```col```×```row```的尺寸读入 与原图尺寸有差异的将自动对图片宽高进行伸缩
#### ```channel```
指定了以什么模式将图片转换成```INDArray``` 该参数取值必须为 1/3/4 中的一个 代表了图片的通道数

* 如果```channel```取值为1 则该方法返回```shape```为```[files.size(), row, col]```的```INDArray``` 该方法会以灰度图模式读取图片

* 如果```channel```取值为3 则该方法返回```shape```为```[files.size(), 3, row, col]```的```INDArray``` 该方法会以RGB模式读取图片<br>
在返回值```[x, 0]```处的二维矩阵为改组图像数据中第```x```张图像的B矩阵<br>
在返回值```[x, 1]```处的二维矩阵为改组图像数据中第```x```张图像的G矩阵<br>
在返回值```[x, 2]```处的二维矩阵为改组图像数据中第```x```张图像的R矩阵

* 如果```channel```取值为4 则该方法返回```shape```为```[files.size(), 4, row, col]```的```INDArray``` 该方法会以ARGB模式读取图片<br>
在返回值```[x, 0]```处的二维矩阵为改组图像数据中第```x```张图像的B矩阵<br>
在返回值```[x, 1]```处的二维矩阵为改组图像数据中第```x```张图像的G矩阵<br>
在返回值```[x, 2]```处的二维矩阵为改组图像数据中第```x```张图像的R矩阵<br>
在返回值```[x, 3]```处的二维矩阵为改组图像数据中第```x```张图像的A矩阵
### 例程
```{java}
INDArray features0 = ImageRead.files2INDArray(files, 4, 300, 300);
INDArray features1 = ImageRead.files2INDArray(files, 4, 300, 300, ImageTran::toGray, ImageTran::inverse);
```
## IdxUbyteRead::fromFile
.idx3-ubyte / .idx1-ubyte是MNIST数据集的文件格式
### 例程
```{java}
INDArray x = IdxUbyteRead.fromFile("F:/Programs/TfDemo/MNIST_data/t10k-images.idx3-ubyte");
INDArray y = IdxUbyteRead.fromFile("F:/Programs/TfDemo/MNIST_data/t10k-labels.idx1-ubyte");
```