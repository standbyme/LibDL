package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.RawDataSet;

import java.util.ArrayList;
import java.util.Collections;

/**
 * 该变换用于打乱{@link RawDataSet}<code>&lt;F,L&gt;</code>
 * @param <F>
 * @param <L>
 */
public class ShuffleRDSTransform<F,L> extends RDSTransform<F,L>{

    /**
     * 传入待处理的{@link RawDataSet} 获得打乱后的{@link RawDataSet}
     * @param oriRawDataSet
     * @return
     */
    @Override
    public RawDataSet<F, L> tran2(RawDataSet<F, L> oriRawDataSet) {
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
