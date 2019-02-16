package com.jstarcraft.module.image2Vec.imagetran;

import javax.swing.*;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.util.function.Consumer;
import java.util.function.Function;

public class ImageTran {

    public static <T> Function<T,T> C2F(Consumer<T> consumer) {
        return t->{consumer.accept(t);return t;};
    }

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

    public static void display(Image image) {
        display(image, "display");
    }

    public static BufferedImage redraw(BufferedImage oriImage, int row, int col, int type) {
        BufferedImage res = new BufferedImage(col, row, type);
        Graphics graphics = res.getGraphics();
        graphics.drawImage(oriImage, 0, 0, col, row, null);
        return res;
    }

    public static BufferedImage toGray(BufferedImage oriImage) {
        new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(oriImage, oriImage);
        return oriImage;
    }

    public static BufferedImage inverse(BufferedImage oriImage) {
        for (int i = 0; i < oriImage.getHeight(); i++) {
            for (int j = 0; j < oriImage.getWidth(); j++) {
                oriImage.setRGB(j, i, 0xffffff ^ oriImage.getRGB(j, i));
            }
        }
        return oriImage;
    }

}