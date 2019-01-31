package com.jstarcraft.module.datatran.labelmaker;

import java.util.ArrayList;

/**
 * 继承自{@link LabelMaker} 并实现了{@link BaseLabelMaker#makeAll(ArrayList)}方法
 * @param <F>
 * @param <L>
 */
public abstract class BaseLabelMaker<F,L> implements LabelMaker<F,L> {
    /**
     * <p>对传入的一组特征对象中每一个特征对象顺序执行{@link LabelMaker#make(Object)}方法</p>
     * @param features 该参数类型应为{@link ArrayList} 表示传入的一组特征对象
     * @return
     */
    @Override
    public ArrayList<L> makeAll(ArrayList<F> features) {
        ArrayList<L> res = new ArrayList<>();
        features.stream().map(this::make).forEach(res::add);
        return res;
    }
}
