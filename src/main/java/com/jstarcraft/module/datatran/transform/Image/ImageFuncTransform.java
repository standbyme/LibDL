package com.jstarcraft.module.datatran.transform.Image;

import com.jstarcraft.module.datatran.transform.FuncTransform;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.function.Function;

public class ImageFuncTransform extends FuncTransform<ArrayList<Mat>> {

    public ImageFuncTransform(Function<Object, ArrayList<Mat>> function) {
        super(function);
    }
}
