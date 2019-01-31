package com.jstarcraft.module.datatran.labelmaker;

import java.util.ArrayList;

/**
 * 顾名思义
 * @param <L>
 */
public interface LabelMaker<F,L> {
    /**
     * 传入一个某类型的特征对象 得到它对应的标签对象
     * @param feature
     * @return
     */
    L make(F feature);
    /**
     * 传入一组某类型的特征对象 得到它对应的一组标签对象
     * @param features
     * @return
     */
    ArrayList<L> makeAll(ArrayList<F> features);
}
