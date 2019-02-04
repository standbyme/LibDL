package com.jstarcraft.module.datatran.function.labelmake;

import com.jstarcraft.module.datatran.entity.FLPair;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 包含和标签生成相关的方法 Label Make这一过程 将一组特征对象转化为一组{@link FLPair}对象
 */
public class LabelMake {
    /**
     * 传入一组特征对象 和由特征对象生成标签对象的方法 返回一组{@link FLPair}对象
     * @param features
     * @param function
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> List<FLPair<F,L>> makeAll(List<F> features, Function<F,L> function) {
        return features.stream().map(f -> new FLPair<>(f,function.apply(f))).collect(Collectors.toCollection(ArrayList::new));
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
