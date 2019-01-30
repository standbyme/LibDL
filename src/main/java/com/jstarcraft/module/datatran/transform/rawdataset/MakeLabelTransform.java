package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.entity.RawDataSet;
import com.jstarcraft.module.datatran.labelmaker.LabelMaker;

import java.util.ArrayList;

public class MakeLabelTransform<F,L> implements RDSTransform<F,L> {

    private LabelMaker<L> labelMaker;

    public MakeLabelTransform(LabelMaker<L> labelMaker){
        this.labelMaker = labelMaker;
    }

    @SuppressWarnings("unchecked")
    @Override
    public RawDataSet tran(Object features) {
        return new RawDataSet((ArrayList)features, labelMaker.makeAll(features));
    }

    public void setLabelMaker(LabelMaker<L> labelMaker) {
        this.labelMaker = labelMaker;
    }

    public LabelMaker<L> getLabelMaker() {
        return labelMaker;
    }
}
