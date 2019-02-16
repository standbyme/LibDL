package com.jstarcraft.module.image2Vec.imageconvert;

import com.jstarcraft.module.image2Vec.imagetran.ImageTran;
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

    @SafeVarargs
    public static INDArray files2INDArray(List<File> files, int channel, int row, int col, Function<BufferedImage, BufferedImage>... functions) {
        return files2INDArray(files, channel, row, col, BufferedImage.TYPE_INT_ARGB, functions);
    }

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

    public static BufferedImage fromFile(File file, int row, int col, int type) {
        try {
            return ImageTran.redraw(ImageIO.read(file), row, col, type);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 如果传入的INDArray不满足图片数据的格式 返回null
     *
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

    public static List<BufferedImage> fromNd4jMatrices(INDArray indArrays) {
        long count = indArrays.shape()[0];
        List<BufferedImage> res = new ArrayList<>();
        for (long i = 0; i < count; i++) {
            INDArray index = Nd4j.create(new double[]{i}, new int[]{1, 1});
            res.add(ImageRead.fromNd4jMatrix(indArrays.get(index)));
        }
        return res;
    }

    public static List<BufferedImage> fromNd4jMatrices(INDArray indArrays, long... indexes) {
        List<BufferedImage> res = new ArrayList<>();
        for (long i = 0; i < indexes.length; i++) {
            INDArray index = Nd4j.create(new double[]{i}, new int[]{1, 1});
            res.add(ImageRead.fromNd4jMatrix(indArrays.get(index)));
        }
        return res;
    }

}
