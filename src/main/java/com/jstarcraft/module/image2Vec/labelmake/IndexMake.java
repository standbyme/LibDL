package com.jstarcraft.module.image2Vec.labelmake;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 将一组标签转化为用于分类的INDArray 例如将数据内容为{"label1","label2","label3","label1"}的{@link List}转化为数据内容为[[1,0,0],[0,1,0],[0,0,1],[1,0,0]]的{@link INDArray}
 */
public class IndexMake {

    /**
     * 返回标签转化后的结果
     *
     * @param labels
     * @param <L>
     * @return
     */
    public static <L> INDArray indexMake(List<L> labels) {
        return indexMake(labels, new HashMap<>());
    }

    /**
     * 将一组特征对象通过指定的方法转换为一组标签 然后转化 返回标签转化后的结果
     *
     * @param features
     * @param function 从特征对象到标签对象的方法
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F, L> INDArray indexMake(List<F> features, Function<F, L> function) {
        List<L> labels = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return indexMake(labels);
    }

    /**
     * @param labels
     * @param label2Index 指定了标签值与索引值的一一对应关系 特别注意的是 当其为空时 该方法运行结束后 该对象中会被添加上默认的对应关系数据
     * @param <L>
     * @return
     */
    public static <L> INDArray indexMake(List<L> labels, Map<L, Integer> label2Index) {
        if (label2Index.isEmpty()) label2Index.putAll(calculateLabel2Index(getDefaultIndex2Label(labels)));
        int[] shape = new int[]{labels.size(), label2Index.size()};
        List<INDArray> labelList = labels.stream().
                map(label2Index::get).
                map(r -> Nd4j.zeros(label2Index.size()).putScalar(r, 1)).
                collect(Collectors.toCollection(ArrayList::new));
        return Nd4j.create(labelList, shape);
    }

    /**
     * @param features
     * @param function
     * @param label2Index 指定了标签值与索引值的一一对应关系 特别注意的是 当其为空时 该方法运行结束后 该对象中会被添加上默认的对应关系数据
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F, L> INDArray indexMake(List<F> features, Function<F, L> function, Map<L, Integer> label2Index) {
        List<L> labels = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return indexMake(labels, label2Index);
    }

    /**
     * @param labels
     * @param index2Label 指定了索引值与标签值的一一对应关系 特别注意的是 当其为空时 该方法运行结束后 该对象中会被添加上默认的对应关系数据
     * @param <L>
     * @return
     */
    public static <L> INDArray indexMake(List<L> labels, List<L> index2Label) {
        if (index2Label.isEmpty()) index2Label.addAll(getDefaultIndex2Label(labels));
        Map<L, Integer> label2Index = calculateLabel2Index(index2Label);
        return indexMake(labels, label2Index);
    }

    /**
     * @param features
     * @param function
     * @param index2Label 指定了索引值与标签值的一一对应关系 特别注意的是 当其为空时 该方法运行结束后 该对象中会被添加上默认的对应关系数据
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F, L> INDArray indexMake(List<F> features, Function<F, L> function, List<L> index2Label) {
        List<L> labels = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return indexMake(labels, index2Label);
    }

    /**
     * 获取默认的索引值与标签值的一一对应关系
     * @param labels
     * @param <L>
     * @return
     */
    public static <L> List<L> getDefaultIndex2Label(List<L> labels) {
        return labels.stream().distinct().collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * 传入指定从索引值到标签值的关系的对象 返回指定从标签值到索引值的关系的对象
     * @param index2Label
     * @param <L>
     * @return
     */
    public static <L> Map<L, Integer> calculateLabel2Index(Collection<L> index2Label) {
        Map<L, Integer> label2Index = new HashMap<>();
        int rank = 0;
        for (L label : index2Label) label2Index.put(label, rank++);
        return label2Index;
    }
}
