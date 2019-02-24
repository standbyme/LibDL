package com.jstarcraft.module.image2Vec.imageconvert;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


/**
 * 将图片对象 即BufferedImage对象中的图片信息写入其他形式的载体中(如文件/INDArray)
 */
public class ImageWrite {

    /**
     * 图片对象写入文件中 文件以创建时的毫秒数命名 所以在创建之前调用了{@link Thread#sleep(long)}暂停1毫秒 防止文件名重复
     * @param image 图片对象
     * @param dirPath 待写入的文件夹路径 该方法会在该文件夹下创建名为"当前系统毫秒数.png"的图片文件 将图片写入
     */
    public static void toDir(BufferedImage image, String dirPath) {
        try {
            Thread.sleep(1);
            dirPath = new File(dirPath).getAbsolutePath();
            File file = new File(dirPath + "/" + System.currentTimeMillis() + ".png");
            file.getParentFile().mkdir();
            ImageIO.write(image, "png", file);
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 将图片对象写入INDArray
     * @param image 图片对象
     * @param channel 指定该图片对象写入INDArray时的通道数
     * @return
     */
    public static INDArray toMatrix(BufferedImage image, int channel) {
        if (!(channel == 1 || channel == 3 || channel == 4)) return Nd4j.create();
        int row = image.getHeight();
        int col = image.getWidth();
        INDArray res = channel == 1 ? Nd4j.create(row, col) : Nd4j.create(channel, row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int argb = image.getRGB(j, i);
                int mask = 0b11111111;
                int b = argb & mask;
                int g = (argb >> 8) & mask;
                int r = (argb >> 16) & mask;
                int a = (argb >> 24) & mask;
                if (channel == 1) {
                    res.putScalar(new int[]{i, j}, (r * 38 + g * 75 + b * 15) >> 7);
                } else {
                    res.putScalar(new int[]{0, i, j}, b);
                    res.putScalar(new int[]{1, i, j}, g);
                    res.putScalar(new int[]{2, i, j}, r);
                    if (channel == 4) {
                        res.putScalar(new int[]{3, i, j}, a);
                    }
                }
            }
        }
        return res;
    }
}
