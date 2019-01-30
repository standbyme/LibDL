package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.entity.RawDataSet;

import java.util.ArrayList;

public class SelectRDSTransform<F,L> implements RDSTransform<F,L>{

    private int lo,hi;

    public SelectRDSTransform(int lo, int hi) {
        this.lo = lo;
        this.hi = hi;
    }

    @SuppressWarnings("unchecked")
    @Override
    public RawDataSet<F, L> tran(Object in) throws Exception {
        RawDataSet<F, L> oriRawDataSet = (RawDataSet<F, L>)in;
        ArrayList<F> oriFeatures = oriRawDataSet.getFeatures();
        ArrayList<L> oriLabels = oriRawDataSet.getLabels();
        ArrayList<F> features = new ArrayList<>(oriFeatures.subList(lo, hi));
        ArrayList<L> labels = new ArrayList<>(oriLabels.subList(lo, hi));
        return new RawDataSet<>(features, labels);
    }
}
