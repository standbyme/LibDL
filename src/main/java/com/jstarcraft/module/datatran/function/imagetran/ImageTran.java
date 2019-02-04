package com.jstarcraft.module.datatran.function.imagetran;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.swing.*;

/**
 * 包含一些对图片对象进行转换的方法
 */
public class ImageTran {
    /**
     * 弹窗展示图片 在JUnit中进行测试的话 弹出的Frame会闪退
     * @param image
     * @param title
     */
    public static void display(Mat image, String title) {
        CanvasFrame canvas = new CanvasFrame(title, 1);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        canvas.showImage(converter.convert(image));
    }

    /**
     * 调用了{@link ImageTran#display(Mat, String)}
     * @param image
     */
    public static void display(Mat image) {
        display(image, "display");
    }
}