package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.entity.RawDataSet;

import java.util.ArrayList;
import java.util.Collections;

public class ShuffleRDSTransform<F,L> implements RDSTransform<F,L>{
    @SuppressWarnings("unchecked")
    @Override
    public RawDataSet<F, L> tran(Object in) {
        RawDataSet<F,L> oriRawDataSet = (RawDataSet<F,L>)in;
        ArrayList<Integer> ranks = new ArrayList<>();
        for (int i=0;i<oriRawDataSet.size();i++)ranks.add(i);
        Collections.shuffle(ranks);
        ArrayList<F> oriFeatures = oriRawDataSet.getFeatures();
        ArrayList<L> oriLabels = oriRawDataSet.getLabels();
        ArrayList<F> shuffledFeatures = new ArrayList<>();
        ArrayList<L> shuffledLabels = new ArrayList<>();
        ranks.forEach(rank->{
            shuffledFeatures.add(oriFeatures.get(rank));
            shuffledLabels.add(oriLabels.get(rank));
        });
        return new RawDataSet<>(shuffledFeatures, shuffledLabels);
    }
}
