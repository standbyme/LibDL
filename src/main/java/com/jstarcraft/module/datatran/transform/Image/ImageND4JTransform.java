package com.jstarcraft.module.datatran.transform.Image;

import com.jstarcraft.module.datatran.transform.BaseTransform;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.datavec.image.loader.NativeImageLoader;

import java.io.IOException;
import java.util.ArrayList;

public class ImageND4JTransform extends BaseTransform<ArrayList<Mat>, ArrayList<INDArray>> {
    @Override
    public ArrayList<INDArray> tran2(ArrayList<Mat> inMats) throws IOException {
        return mats2INDArray(inMats);
    }

    public ArrayList<INDArray> mats2INDArray(ArrayList<Mat> mats) throws IOException {
        if (mats.size() <= 0) return null;
        NativeImageLoader imageLoader = new NativeImageLoader();
        ArrayList<INDArray> indArrays = new ArrayList<>();
        for (Mat mat : mats)indArrays.add(imageLoader.asMatrix(mat));
        return indArrays;
    }
}
