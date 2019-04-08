package vision.datasets._ImageModule.image2Vec.imageconvert;

import vision.datasets._ImageModule.image2Vec.imagetran.ImageTran;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 从数据来源(如{@link File}对象) 获取图片对象
 */
public class ImageRead {
    /**
     * 将一组图像文件转化为INDArray数据集
     * @param files
     * @param channel 指定该图片对象<b>写入</b>INDArray时的通道数 该参数取值必须为 1/3/4 中的一个 代表了图片的通道数<ul><li>如果channel取值为1 则该方法返回shape为[files.size(), row, col]的INDArray 该方法会以灰度图模式读取图片</li><li>如果channel取值为3 则该方法返回shape为[files.size(), 3, row, col]的INDArray 该方法会以RGB模式读取图片<br>在返回值[x, 0]处的二维矩阵为改组图像数据中第x张图像的B矩阵<br>在返回值[x, 1]处的二维矩阵为改组图像数据中第x张图像的G矩阵<br>在返回值[x, 2]处的二维矩阵为改组图像数据中第x张图像的R矩阵</li><li>如果channel取值为4 则该方法返回shape为[files.size(), 4, row, col]的INDArray 该方法会以ARGB模式读取图片<br>在返回值[x, 0]处的二维矩阵为改组图像数据中第x张图像的B矩阵<br>在返回值[x, 1]处的二维矩阵为改组图像数据中第x张图像的G矩阵<br>在返回值[x, 2]处的二维矩阵为改组图像数据中第x张图像的R矩阵<br>在返回值[x, 3]处的二维矩阵为改组图像数据中第x张图像的A矩阵</li></ul>
     * @param row 将该组图片以宽×高为col×row的尺寸<b>读入</b> 与原图尺寸有差异的将自动对图片宽高进行伸缩
     * @param col 将该组图片以宽×高为col×row的尺寸<b>读入</b> 与原图尺寸有差异的将自动对图片宽高进行伸缩
     * @param functions 该方法的读取过程是 先从文件中读图像数据到BufferedImage对象 然后再将读取到的一组BufferedImage对象转换成INDArray<br> 在从文件中读图像数据到BufferedImage对象之后 将读取到的一组BufferedImage对象转换成INDArray之前 该方法将对每个BufferedImage对象按照数组顺序执行functions中储存的操作
     * @return
     */
    @SafeVarargs
    public static INDArray files2INDArray(List<File> files, int channel, int row, int col, Function<BufferedImage, BufferedImage>... functions) {
        return files2INDArray(files, channel, row, col, BufferedImage.TYPE_INT_ARGB, functions);
    }

    /**
     * 将一组图像文件转化为INDArray数据集 并指定了读入文件时的图片对象的type
     * @param files
     * @param channel
     * @param row
     * @param col
     * @param type 将该组图片以指定的type<b>读入</b> type的取值参见https://docs.oracle.com/javase/6/docs/api/constant-values.html#java.awt.image.BufferedImage.TYPE_INT_BGR
     * @param functions
     * @return
     */
    @SafeVarargs
    public static INDArray files2INDArray(List<File> files, int channel, int row, int col, int type, Function<BufferedImage, BufferedImage>... functions) {
        Stream<BufferedImage> stream = files.stream().
                map(f -> ImageRead.fromFile(f, row, col, type));

        for (Function<BufferedImage,BufferedImage> function : functions) {
            stream = stream.map(function);
        }

        List<INDArray> featureList = stream.map(bi -> ImageWrite.toMatrix(bi, channel)).
                collect(Collectors.toCollection(ArrayList::new));
        int[] shape = channel==1?new int[]{files.size(), row, col}:new int[]{files.size(), channel, row, col};
        return Nd4j.create(featureList, shape);
    }

    /**
     * 将一个图像文件转化为一个BufferedImage对象
     * @param file
     * @param row 将该图片以宽×高为col×row的尺寸读入 与原图尺寸有差异的将自动对图片宽高进行伸缩
     * @param col 将该图片以宽×高为col×row的尺寸读入 与原图尺寸有差异的将自动对图片宽高进行伸缩
     * @param type 将该组图片以指定的type<b>读入</b> type的取值参见https://docs.oracle.com/javase/6/docs/api/constant-values.html#java.awt.image.BufferedImage.TYPE_INT_BGR
     * @return
     */
    public static BufferedImage fromFile(File file, int row, int col, int type) {
        try {
            return ImageTran.redraw(ImageIO.read(file), row, col, type);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 传入含有一张图片的数据的INDArray对象 转化为一个BufferedImage对象 如果传入的INDArray不满足图片数据的格式 返回null 关于图片数据在INDArray中的格式 参见{@link ImageRead#files2INDArray}中的对channel变量的解释
     * @param indArray
     * @return
     */
    public static BufferedImage fromNd4jMatrix(INDArray indArray) {
        long[] shape = indArray.shape();
        int ndim = shape.length;
        if (ndim != 3 && ndim != 2) return null;
        int channel = ndim == 3 ? (int) shape[0] : 1;
        int row = (int) shape[ndim - 2];
        int col = (int) shape[ndim - 1];

        BufferedImage oriImage = new BufferedImage(col, row, BufferedImage.TYPE_INT_ARGB);

        long area = row * col;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int b = channel == 1 ? indArray.getInt(i, j) : indArray.getInt(0, i, j);
                int g = channel > 1 ? indArray.getInt(1, i, j) : b;
                int r = channel > 2 ? indArray.getInt(2, i, j) : b;
                int a = channel > 3 ? indArray.getInt(3, i, j) : 0xff;
                try {
                    oriImage.setRGB(j, i, new Color(r, g, b, a).getRGB());
                } catch (IllegalArgumentException e) {
                    System.out.println(r + " " + g + " " + b + " " + a);
                    throw e;
                }
            }
        }

        int type;
        if (channel == 1) type = BufferedImage.TYPE_BYTE_GRAY;
        else if (channel == 3) type = BufferedImage.TYPE_INT_RGB;
        else if (channel == 4) type = BufferedImage.TYPE_INT_ARGB;
        else return null;

        return ImageTran.redraw(oriImage, row, col, type);
    }

    /**
     * 传入含有一组图片的数据的INDArray对象 转化为一组BufferedImage对象
     * @param indArrays
     * @return
     */
    public static List<BufferedImage> fromNd4jMatrices(INDArray indArrays) {
        long count = indArrays.shape()[0];
        List<BufferedImage> res = new ArrayList<>();
        for (long i = 0; i < count; i++) {
            INDArray index = Nd4j.create(new double[]{i}, new int[]{1, 1});
            res.add(ImageRead.fromNd4jMatrix(indArrays.get(index)));
        }
        return res;
    }

    /**
     * 传入含有一组图片的数据的INDArray对象 并指定了需要抽取的部分图片的索引值 将对应索引值的一组图片数据 转化为一组BufferedImage对象
     * @param indArrays
     * @param indexes
     * @return
     */
    public static List<BufferedImage> fromNd4jMatrices(INDArray indArrays, long... indexes) {
        List<BufferedImage> res = new ArrayList<>();
        for (int i = 0; i < indexes.length; i++) {
            INDArray index = Nd4j.create(new double[]{indexes[i]}, new int[]{1, 1});
            res.add(ImageRead.fromNd4jMatrix(indArrays.get(index)));
        }
        return res;
    }

}
