package com.jstarcraft.module.datatran.transform.Image;

import com.jstarcraft.module.datatran.entity.RawDataSet;
import com.jstarcraft.module.datatran.transform.Transform;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.datavec.image.loader.NativeImageLoader;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;

public class ImageND4JTransform implements Transform<DataSet> {
    @Override
    public DataSet tran(Object in) throws IOException {
        RawDataSet<Mat,Object> inRawDataSet = (RawDataSet<Mat, Object>)in;
        ArrayList<Mat> inMats = inRawDataSet.getFeatures();
        ArrayList<Object> inLabels = inRawDataSet.getLabels();
        return new DataSet(mats2INDArray(inMats), labelsVStack(inLabels));
    }

    private INDArray labelsVStack(ArrayList<Object> labels) {
        if (labels.size() <= 0) return null;
        INDArray[] indArrays = new INDArray[labels.size()];
        for (int i=0;i<labels.size();i++)indArrays[i] = toINDArray(labels.get(i));
        return Nd4j.vstack(indArrays);
    }

    private INDArray toINDArray(Object label) {
        if (label instanceof Number) return Nd4j.ones(1,1).muli((Number) label);
        else return (INDArray)label;
    }

    public INDArray mats2INDArray(ArrayList<Mat> mats) throws IOException {
        if (mats.size() <= 0) return null;
        NativeImageLoader imageLoader = new NativeImageLoader();
        INDArray[] indArrays = new INDArray[mats.size()];
        for (int i=0;i<mats.size();i++)indArrays[i] = imageLoader.asMatrix(mats.get(i));
        return Nd4j.vstack(indArrays);
    }
}
