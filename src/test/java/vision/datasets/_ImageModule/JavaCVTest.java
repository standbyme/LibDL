package vision.datasets._ImageModule;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_core.flip;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.circle;

/**
 * JavaCV工具类
 */
class JavaCVUtil {
    /**
     * 功能说明:显示图像
     *
     * @param mat   要显示的mat类型图像
     * @param title 窗口标题
     */
    public static void imShow(Mat mat, String title) {
        //opencv自带的显示模块，跨平台性欠佳，转为Java2D图像类型进行显示
        ToMat converter = new OpenCVFrameConverter.ToMat();
        CanvasFrame canvas = new CanvasFrame(title, 1);
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        canvas.showImage(converter.convert(mat));
    }

    /**
     * 功能说明:保存mat到指定路径
     *
     * @param mat      要保存的Mat
     * @param filePath 保存路径
     */
    public static boolean imWrite(Mat mat, String filePath) {
        //不包含中文，直接使用opencv原生方法进行保存
        if (!containChinese(filePath)) {
            return opencv_imgcodecs.imwrite(filePath, mat);
        }
        try {
            /*
             * 将mat转为java的BufferedImage
             */
            ToMat convert = new ToMat();
            Frame frame = convert.convert(mat);
            Java2DFrameConverter java2dFrameConverter = new Java2DFrameConverter();
            BufferedImage bufferedImage = java2dFrameConverter.convert(frame);
            ImageIO.write(bufferedImage, "PNG", new File(filePath));
            return true;
        } catch (Exception e) {
            System.out.println("保存文件出现异常:" + filePath);
            e.printStackTrace();
        }
        return false;
    }

    /**
     * 功能说明:判断字符是否包含中文
     *
     * @param inputString
     * @return boolean
     */
    private static boolean containChinese(String inputString) {
        //四段范围，包含全面
        String regex = "[\\u4E00-\\u9FA5\\u2E80-\\uA4CF\\uF900-\\uFAFF\\uFE30-\\uFE4F]";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(inputString);
        return matcher.find();
    }
}

public class JavaCVTest {
    public static void main(String[] args) {
        //以彩色模式读取图像
        Mat image = imread("C:/Index/Program/datatran/img/label1/1.jpg", IMREAD_COLOR);
        if (image == null || image.empty()) {
            System.out.println("读取图像失败，图像为空");
            return;
        }
        System.out.println("图像宽x高" + image.cols() + " x " + image.rows());
        /*
         * 显示图像,opencv自带的显示方法，跨平台性能不好，转换为java2D显示图像
         * windows下可以使用如下代码进行显示
         * opencv_highgui.imshow("原始图像", image);
         */
        JavaCVUtil.imShow(image, "原始图像");
        //创建空mat，保存处理图像
        Mat result = new Mat();
        int flipCode = 1;
        /*
         * flipCode
         * >0	水平翻转
         * =0	垂直翻转
         * <0	同时翻转
         *
         */
        flip(image, result, flipCode);
        //显示处理过的图像
        JavaCVUtil.imShow(result, "水平翻转");
        /*
         * 保存图像
         * 也可使用opencv原生方法 opencv_imgcodecs. imwrite("output.bmp", result);
         */
        JavaCVUtil.imWrite(result, "data/javacv/lakeResult.jpg");
        //克隆图像
        Mat imageCircle = image.clone();
        /*
         * 在图像上画圆
         */
        circle(imageCircle, // 目标图像
                new Point(420, 150), // 圆心坐标
                65, // radius
                new Scalar(0, 200, 0, 0), // 颜色，绿色
                2, // 线宽
                8, // 8-connected line
                0); // shift
        opencv_imgproc.putText(imageCircle, //目标图像
                "Lake and Tower", // 文本内容(不可包含中文)
                new Point(460, 200), // 文本起始位置坐标
                FONT_HERSHEY_PLAIN, // 字体类型
                2.0, // 字号大小
                new Scalar(0, 255, 0, 3), //文本颜色，绿色
                1, // 文本字体线宽
                8, // 线形.
                false); //控制文本走向
        JavaCVUtil.imShow(imageCircle, "画圆mark");
    }
}
