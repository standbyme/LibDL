package com.jstarcraft.module.datatran.function.imageload;

import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 将javaCV(openCV)的图片对象转换为ND4J的格式 该类中的方法 暂时都还比较粗犷
 */
public class Image2Nd4j {
    /**
     * 将图片对象转换为多个矩阵的叠加形式 返回一个四维{@link INDArray} 其维度为[1, 图片通道数, 行数, 列数]
     * @param image
     * @return
     */
    public static INDArray asMatrix(Mat image) {
        int channels = image.channels(), rows = image.rows(), cols = image.cols();
        INDArray ret = Nd4j.create(1,channels, rows, cols);

        Indexer idx = image.createIndexer();

        for (int k=0;k<channels;k++) {
            for (int i=0;i<rows;i++) {
                for (int j=0;j<cols;j++) {
                    ret.put(new int[]{0,k,i,j},Nd4j.ones(1,1).mul(idx.getDouble(i,j,k)));
                }
            }
        }
        return ret;
    }
}
