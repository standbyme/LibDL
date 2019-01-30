package com.jstarcraft.module.datatran.transform.Image;

import com.jstarcraft.module.datatran.transform.Transform;
import org.bytedeco.javacpp.opencv_core.Mat;

import java.io.File;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;

public class ImageLoadTransform implements Transform<ArrayList<Mat>> {
    @SuppressWarnings("unchecked")
    @Override
    public ArrayList<Mat> tran(Object in) {
        ArrayList<File> inFiles = (ArrayList<File>) in;
        ArrayList<Mat> outMats = new ArrayList<>();
        inFiles.stream().map(f -> {
            return imread(f.getAbsolutePath(), IMREAD_COLOR);
        }).forEach(outMats::add);
        return outMats;
    }
}
