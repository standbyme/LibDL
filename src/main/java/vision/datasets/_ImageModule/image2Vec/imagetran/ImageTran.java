package vision.datasets._ImageModule.image2Vec.imagetran;

import javax.swing.*;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * 内含一些对图像对象的转换方法
 */
public class ImageTran {
    /**
     * 将{@link Consumer}&lt;T&gt;对象转换为返回原传入值的{@link Function}&lt;T,T&gt;对象
     * @param consumer
     * @param <T>
     * @return
     */
    public static <T> Function<T,T> C2F(Consumer<T> consumer) {
        return t->{consumer.accept(t);return t;};
    }

    /**
     * 展示图片
     * @param image 待展示的图片对象
     * @param title 展示时的标题
     */
    public static void display(final Image image, String title) {
        JPanel jPanel = new JPanel(){
            @Override
            public void paintComponent(Graphics g) {
                g.drawImage(image, 0, 0, null);
            }
        };

        JFrame jframe = new JFrame(title);
        jframe.add(jPanel);
        jframe.setVisible(true);
        jframe.setSize(500, 500);
        jframe.setBackground(Color.BLUE);
        jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    /**
     * 展示图片 展示时的标题为默认的"display"
     * @param image
     */
    public static void display(Image image) {
        display(image, "display");
    }

    /**
     * 将图片对象重新绘制为宽度为col 高度为row 类型为type的图片对象
     * @param oriImage
     * @param row
     * @param col
     * @param type
     * @return
     */
    public static BufferedImage redraw(BufferedImage oriImage, int row, int col, int type) {
        BufferedImage res = new BufferedImage(col, row, type);
        Graphics graphics = res.getGraphics();
        graphics.drawImage(oriImage, 0, 0, col, row, null);
        return res;
    }

    /**
     * 将图片转化为灰度图
     * @param oriImage
     * @return
     */
    public static BufferedImage toGray(BufferedImage oriImage) {
        new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(oriImage, oriImage);
        return oriImage;
    }

    /**
     * 将图片反相
     * @param oriImage
     * @return
     */
    public static BufferedImage inverse(BufferedImage oriImage) {
        for (int i = 0; i < oriImage.getHeight(); i++) {
            for (int j = 0; j < oriImage.getWidth(); j++) {
                oriImage.setRGB(j, i, 0xffffff ^ oriImage.getRGB(j, i));
            }
        }
        return oriImage;
    }

}