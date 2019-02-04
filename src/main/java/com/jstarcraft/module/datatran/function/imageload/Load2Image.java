package com.jstarcraft.module.datatran.function.imageload;

import org.bytedeco.javacpp.opencv_core.Mat;

import java.io.File;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;

/**
 * 从数据来源(如{@link File}对象) 获取javaCV(openCV)的{@link Mat}图片对象
 */
public class Load2Image {
    /**
     * 从{@link File}对象中获取图片对象
     * @param file
     * @return
     */
    public static Mat fromFile(File file) {
        return imread(file.getAbsolutePath(), IMREAD_COLOR);
    }
}
