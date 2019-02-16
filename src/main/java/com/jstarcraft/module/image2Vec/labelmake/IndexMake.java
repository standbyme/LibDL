package com.jstarcraft.module.image2Vec.labelmake;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class IndexMake {

    public static <L> INDArray indexMake(List<L> labels, Map<L, Integer> label2Index) {
        if (label2Index.isEmpty()) label2Index.putAll(calculateLabel2Index(getDefaultIndex2Label(labels)));
        int[] shape = new int[]{labels.size(), label2Index.size()};
        List<INDArray> labelList = labels.stream().
                map(label2Index::get).
                map(r -> Nd4j.zeros(label2Index.size()).putScalar(r, 1)).
                collect(Collectors.toCollection(ArrayList::new));
        return Nd4j.create(labelList, shape);
    }

    public static <F,L> INDArray indexMake(List<F> features, Function<F,L> function, Map<L, Integer> label2Index) {
        List<L> labels = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return indexMake(labels, label2Index);
    }

    public static <L> INDArray indexMake(List<L> labels, List<L> index2Label) {
        if (index2Label.isEmpty()) index2Label.addAll(getDefaultIndex2Label(labels));
        Map<L, Integer> label2Index = calculateLabel2Index(index2Label);
        return indexMake(labels, label2Index);
    }

    public static <F,L> INDArray indexMake(List<F> features, Function<F,L> function, List<L> index2Label) {
        List<L> labels = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return indexMake(labels, index2Label);
    }

    public static <L> INDArray indexMake(List<L> labels) {
        return indexMake(labels, new HashMap<>());
    }

    public static <F,L> INDArray indexMake(List<F> features, Function<F,L> function) {
        List<L> labels = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return indexMake(labels);
    }

    public static <L> List<L> getDefaultIndex2Label(List<L> labels) {
        return labels.stream().distinct().collect(Collectors.toCollection(ArrayList::new));
    }

    public static <L> Map<L, Integer> calculateLabel2Index(Collection<L> index2Label) {
        Map<L, Integer> label2Index = new HashMap<>();
        int rank = 0;
        for (L label : index2Label) label2Index.put(label, rank++);
        return label2Index;
    }
}
