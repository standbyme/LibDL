package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.entity.RawDataSet;
import com.jstarcraft.module.datatran.transform.MultiTransform;
import com.jstarcraft.module.datatran.transform.Transform;

import java.util.ArrayList;

public class BranchTransform<F,L> implements RDSTransform<F,L>{
    private MultiTransform<ArrayList<F>> featuresTransform;
    private MultiTransform<ArrayList<L>> labelsTransform;

    public BranchTransform() {
        this(new MultiTransform<>(), new MultiTransform<>());
    }

    public BranchTransform(MultiTransform<ArrayList<F>> featuresTransform, MultiTransform<ArrayList<L>> labelsTransform) {
        this.featuresTransform = featuresTransform;
        this.labelsTransform = labelsTransform;
    }

    public void addFTransform(Transform... transforms) {
        featuresTransform.add(transforms);
    }

    public void addLTransform(Transform... transforms) {
        labelsTransform.add(transforms);
    }

    @SuppressWarnings("unchecked")
    @Override
    public RawDataSet<F, L> tran(Object in) throws Exception{
        RawDataSet inRawDataSet = (RawDataSet)in;
        ArrayList oriFeatures = inRawDataSet.getFeatures();
        ArrayList oriLabels = inRawDataSet.getLabels();

        ArrayList<F> features;
        if (featuresTransform == null) features = (ArrayList<F>)oriFeatures;
        else features = featuresTransform.tran(oriFeatures);

        ArrayList<L> labels;
        if (labelsTransform == null) labels = (ArrayList<L>)oriLabels;
        else labels = labelsTransform.tran(oriLabels);

        return new RawDataSet<>(features, labels);
    }

    public void setFeaturesTransform(MultiTransform<ArrayList<F>> featuresTransform) {
        this.featuresTransform = featuresTransform;
    }

    public void setLabelsTransform(MultiTransform<ArrayList<L>> labelsTransform) {
        this.labelsTransform = labelsTransform;
    }

    public MultiTransform<ArrayList<F>> getFeaturesTransform() {
        return featuresTransform;
    }

    public MultiTransform<ArrayList<L>> getLabelsTransform() {
        return labelsTransform;
    }
}
