package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.entity.RawDataSet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class BalanceLabelTransform<F, L> implements RDSTransform<F,L> {

    @SuppressWarnings("unchecked")
    @Override
    public RawDataSet<F, L> tran(Object in) {
        RawDataSet<F, L> oriDataSet = (RawDataSet<F, L>) in;
        ArrayList<L> oriLabels = oriDataSet.getLabels();
        ArrayList<F> oriFeatures = oriDataSet.getFeatures();
        HashMap<L, ArrayList<Integer>> staMap = new HashMap<>();

        int rank = 0;
        for (L label : oriLabels) {
            ArrayList<Integer> staL = staMap.computeIfAbsent(label, k->new ArrayList<>());
            staL.add(rank ++);
        }

        Integer min = staMap.values().stream()
                .mapToInt(ArrayList::size).min().orElse(0);

        ArrayList<L> balLabels = new ArrayList<>();
        ArrayList<F> balFeatures = new ArrayList<>();

        staMap.values().forEach(arr->{
            Collections.shuffle(arr);
            int m = min;
            for(Integer r : arr) {
                balFeatures.add(oriFeatures.get(r));
                balLabels.add(oriLabels.get(r));
                if (--m == 0)break;
            }
        });
        return new RawDataSet<>(balFeatures, balLabels);
    }
}
