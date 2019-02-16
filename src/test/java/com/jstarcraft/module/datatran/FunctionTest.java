package com.jstarcraft.module.datatran;

import com.jstarcraft.module.idxUbyte2Vec.IdxUbyteRead;
import com.jstarcraft.module.image2Vec.flatroot.FlatRoot;
import com.jstarcraft.module.image2Vec.imageconvert.ImageRead;
import com.jstarcraft.module.image2Vec.imagetran.ImageTran;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.List;

public class FunctionTest {


    public static void demo(long a1) {
        System.out.println("demo0");
    }

    public static void demo(long a1, int... a2) {
        System.out.println("demo1");
    }

    public static void main(String[] args) {

        String[] allowedExtensions = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};
        INDArray y = IdxUbyteRead.fromFile("F:/Programs/TfDemo/MNIST_data/t10k-labels.idx1-ubyte");
        INDArray x = IdxUbyteRead.fromFile("F:/Programs/TfDemo/MNIST_data/t10k-images.idx3-ubyte");
//        ImageRead.fromNd4jMatrices(x, 128, 256, 1024).forEach(ImageTran::display);

//        List<BufferedImage> images = ImageRead.fromNd4jMatrices(x);
//        for (int i = 0; i < images.size(); i++) {
//            ImageWrite.toDir(images.get(i), "F:/Programs/TfDemo/MNIST_data/ori/"+y.getInt(i));
//        }


        List<File> files = FlatRoot.fileFR("F:/Programs/TfDemo/MNIST_data/ori/", allowedExtensions);
//
        INDArray features0 = ImageRead.files2INDArray(files, 1, 28, 28, ImageTran::inverse);
        System.out.println(features0);
        // INDArray features1 = ImageRead.files2INDArray(files, 4, 300, 300, ImageTran::toGray, ImageTran::inverse);


//        int row = 2, col = 2, channel = 3;
//        List<File> files = FlatRoot.fileFR("F:/Programs/moduleimage/ImageDemo", allowedExtensions);
//
//        INDArray features = ImageRead.files2INDArray(files, channel, row, col);
//        System.out.println(features);
//
//        IdxUbyteWrite.toFile(features, "F:/Programs/moduleimage/out/idx", 2048);
//        INDArray res = IdxUbyteRead.fromFile("F:/Programs/moduleimage/out/idx");
//        //System.out.println(res);
//
//        INDArray labels = IndexMake.indexMake(files, LabelMake::parentPathLM);
//        System.out.println(labels);
    }
}
