package com.jstarcraft.module.image2Vec.labelmake;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 包含和标签生成相关的方法
 */
public class LabelMake {
    public static <F> INDArray labelMake(List<F> features, Function<F,INDArray> function, int...shape) {
        ArrayList<INDArray> indArrays = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return Nd4j.create(indArrays, shape);
    }

    /**
     * 从{@link File}对象获得它所在路径的名称 作为该对象的标签 该方法是由特征对象生成标签对象的方法之一
     * @param feature
     * @return
     */
    public static String parentPathLM(File feature) {
        return feature.getParentFile().getName();
    }
}
