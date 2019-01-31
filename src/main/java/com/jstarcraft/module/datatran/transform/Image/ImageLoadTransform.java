package com.jstarcraft.module.datatran.transform.Image;

import com.jstarcraft.module.datatran.transform.BaseTransform;
import org.bytedeco.javacpp.opencv_core.Mat;

import java.io.File;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;

public class ImageLoadTransform extends BaseTransform<ArrayList<File>, ArrayList<Mat>> {

    @Override
    public ArrayList<Mat> tran2(ArrayList<File> inFiles) {
        ArrayList<Mat> outMats = new ArrayList<>();
        inFiles.stream().map(f -> {
            return imread(f.getAbsolutePath(), IMREAD_COLOR);
        }).forEach(outMats::add);
        return outMats;
    }
}
