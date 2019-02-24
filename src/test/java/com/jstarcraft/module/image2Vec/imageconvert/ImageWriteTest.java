package com.jstarcraft.module.image2Vec.imageconvert;

import com.jstarcraft.module.image2Vec.imagetran.ImageTran;
import org.junit.Test;

import java.awt.image.BufferedImage;
import java.io.File;

public class ImageWriteTest {
    @Test
    public void toDirTest() {
        String path = "F:/Programs/moduleimage/ImageDemo/label1/1.jpg";
        BufferedImage bufferedImage = ImageRead.fromFile(new File(path), 50, 50, BufferedImage.TYPE_INT_ARGB);

        ImageTran.display(bufferedImage);

        ImageWrite.toDir(bufferedImage, "F:/Programs/moduleimage/out");
    }
}
