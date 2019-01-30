package com.jstarcraft.module.datatran.transform.pretreatment;

import com.jstarcraft.module.datatran.transform.Transform;

import java.util.*;

public class IndexTransform implements Transform<ArrayList<Integer>> {
    private List<?> index2Label;

    public IndexTransform() {
        this.index2Label = new ArrayList<>();
    }

    public IndexTransform(List<?> index2Label) {
        this.index2Label = index2Label;
    }

    @SuppressWarnings("unchecked")
    @Override
    public ArrayList<Integer> tran(Object in) {
        ArrayList<?> rawLabels = (ArrayList) in;

        if (index2Label.isEmpty()) {
            index2Label.addAll(getDefaultIndex2Label(rawLabels));
        }

        Map<Object, Integer> label2Index = new HashMap<>();
        int rank = 0;
        for (Object label : index2Label) label2Index.put(label, rank++);

        ArrayList<Integer> indexedLabels = new ArrayList<>();
        rawLabels.forEach(l -> indexedLabels.add(label2Index.get(l)));

        return indexedLabels;
    }

    public List getDefaultIndex2Label(List rawLabels) {
        return Arrays.asList(rawLabels.stream().distinct().toArray());
    }
}
