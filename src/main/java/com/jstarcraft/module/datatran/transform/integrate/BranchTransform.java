package com.jstarcraft.module.datatran.transform.integrate;

import com.jstarcraft.module.datatran.RawDataSet;
import com.jstarcraft.module.datatran.transform.BaseTransform;
import com.jstarcraft.module.datatran.transform.Transform;

import java.util.ArrayList;

public class BranchTransform<IF,IL,OF,OL> extends BaseTransform<RawDataSet<IF,IL>, RawDataSet<OF,OL>> {
    private MultiTransform<ArrayList<IF>,ArrayList<OF>> featuresTransform;
    private MultiTransform<ArrayList<IL>,ArrayList<OL>> labelsTransform;

    public BranchTransform() {
        this(new MultiTransform<>(), new MultiTransform<>());
    }

    public BranchTransform(MultiTransform<ArrayList<IF>,ArrayList<OF>> featuresTransform, MultiTransform<ArrayList<IL>,ArrayList<OL>> labelsTransform) {
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
    public RawDataSet<OF, OL> tran2(RawDataSet<IF, IL> inRawDataSet) throws Exception{
        ArrayList oriFeatures = inRawDataSet.getFeatures();
        ArrayList oriLabels = inRawDataSet.getLabels();

        ArrayList<OF> features;
        if (featuresTransform == null) features = (ArrayList<OF>)oriFeatures;
        else features = featuresTransform.tran(oriFeatures);

        ArrayList<OL> labels;
        if (labelsTransform == null) labels = (ArrayList<OL>)oriLabels;
        else labels = labelsTransform.tran(oriLabels);

        return new RawDataSet<>(features, labels);
    }

    public MultiTransform<ArrayList<IF>, ArrayList<OF>> getFeaturesTransform() {
        return featuresTransform;
    }

    public MultiTransform<ArrayList<IL>, ArrayList<OL>> getLabelsTransform() {
        return labelsTransform;
    }

    public void setLabelsTransform(MultiTransform<ArrayList<IL>, ArrayList<OL>> labelsTransform) {
        this.labelsTransform = labelsTransform;
    }

    public void setFeaturesTransform(MultiTransform<ArrayList<IF>, ArrayList<OF>> featuresTransform) {
        this.featuresTransform = featuresTransform;
    }
}
