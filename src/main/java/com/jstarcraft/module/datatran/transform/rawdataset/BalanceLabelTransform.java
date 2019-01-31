package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.RawDataSet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * 用于平衡{@link RawDataSet}中各个标签对应的数据条数 经过该变换得到的{@link RawDataSet} 各个标签对应的数据条数相等
 * @param <F>
 * @param <L>
 */
public class BalanceLabelTransform<F, L> extends RDSTransform<F,L> {
    /**
     * 传入原{@link RawDataSet}变量 返回经过删减后变得平衡了的{@link RawDataSet}变量
     * @param oriDataSet
     * @return
     */
    @Override
    public RawDataSet<F, L> tran2(RawDataSet<F, L> oriDataSet) {
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
