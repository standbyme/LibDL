package vision.datasets._ImageModule.image2Vec.imageconvert;

import vision.datasets._ImageModule.image2Vec.imagetran.ImageTran;

import java.awt.image.BufferedImage;
import java.io.File;

public class ImageWriteTest {

    public void toDirTest() {
        String path = "F:/Programs/moduleimage/ImageDemo/label1/1.jpg";
        BufferedImage bufferedImage = ImageRead.fromFile(new File(path), 50, 50, BufferedImage.TYPE_INT_ARGB);

        ImageTran.display(bufferedImage);

        ImageWrite.toDir(bufferedImage, "F:/Programs/moduleimage/out");
    }
}
