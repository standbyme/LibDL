package vision.datasets._ImageModule.image2Vec.imageconvert;

import vision.datasets._ImageModule.image2Vec.flatroot.FlatRoot;
import vision.datasets._ImageModule.image2Vec.imagetran.ImageTran;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.List;

public class ImageReadTest {

    public void files2INDArrayTest(){
        // 注意 在Test类中 display方法弹出的窗口会闪退
        String[] allowedExtensions = new String[]{"jpg"};
        List<File> files = FlatRoot.fileFR("F:/Programs/moduleimage/ImageDemo", allowedExtensions);
        INDArray res = ImageRead.files2INDArray(files, 4, 50, 50, ImageTran::inverse, ImageTran.C2F(ImageTran::display));
        System.out.println(res);
    }
}
