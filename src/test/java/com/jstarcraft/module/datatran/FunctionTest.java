package com.jstarcraft.module.datatran;

import com.jstarcraft.module.datatran.function.flatroot.FlatRoot;
import com.jstarcraft.module.datatran.function.imageload.Image2Nd4j;
import com.jstarcraft.module.datatran.function.imageload.Load2Image;
import com.jstarcraft.module.datatran.function.imagetran.ImageTran;
import com.jstarcraft.module.datatran.function.labelmake.LabelMake;
import com.jstarcraft.module.datatran.translator.Translator;
import org.nd4j.linalg.dataset.DataSet;

public class FunctionTest {

    private static String[] allowedExtensions = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};

    public static void main(String[] args) {
        Translator.init(FlatRoot.fileFR("C:/Index/Program/datatran/img",allowedExtensions),LabelMake::parentPathLM)
                .balanceLabel().shuffle().indexMake().stream().mapF(Load2Image::fromFile)
                .collect2Translator().sample(2).forEachF(ImageTran::display);

        DataSet dataSet = Translator.asDataSet(
                Translator.init(FlatRoot.fileFR("C:/Index/Program/datatran/img",allowedExtensions),LabelMake::parentPathLM)
                        .balanceLabel().shuffle().indexMake().stream()
                        .mapF(Load2Image::fromFile).mapF(Image2Nd4j::asMatrix)
                        .collect2Translator()
        );
        System.out.println(dataSet);
    }
}
