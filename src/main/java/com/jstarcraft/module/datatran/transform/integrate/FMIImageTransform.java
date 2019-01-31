package com.jstarcraft.module.datatran.transform.integrate;

import com.jstarcraft.module.datatran.transform.BaseTransform;
import com.jstarcraft.module.datatran.transform.Image.ImageLoadTransform;
import com.jstarcraft.module.datatran.transform.Image.ImageND4JTransform;
import com.jstarcraft.module.datatran.transform.Transform;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.ArrayList;

public class FMIImageTransform extends BaseTransform<ArrayList<File>, ArrayList<INDArray>> {

    private ImageLoadTransform imageLoadTransform = new ImageLoadTransform();
    private ImageND4JTransform imageND4JTransform = new ImageND4JTransform();
    private MultiTransform<ArrayList<Mat>, ArrayList<Mat>> matTransforms = new MultiTransform<>();

    public FMIImageTransform(Transform... transforms) {
        add(transforms);
    }

    @Override
    public ArrayList<INDArray> tran(Object in) throws Exception {
        ArrayList<Mat> matsStart = imageLoadTransform.tran(in);
        ArrayList<Mat> matsEnd = matTransforms.tran(matsStart);
        return imageND4JTransform.tran(matsEnd);
    }

    @Override
    public Object tran2(ArrayList<File> files) throws Exception {
        return null;
    }

    public void add(Transform... transforms) {
        matTransforms.add(transforms);
    }

    public MultiTransform<ArrayList<Mat>, ArrayList<Mat>> getMatTransforms() {
        return matTransforms;
    }

    public void setMatTransforms(MultiTransform<ArrayList<Mat>, ArrayList<Mat>> matTransforms) {
        this.matTransforms = matTransforms;
    }
}
