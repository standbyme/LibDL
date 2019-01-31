package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.RawDataSet;
import com.jstarcraft.module.datatran.labelmaker.LabelMaker;
import com.jstarcraft.module.datatran.transform.BaseTransform;

import java.util.ArrayList;

/**
 * 原始数据为一组特征 使用{@link MakeLabelTransform#labelMaker}生成特征对应的一组标签 得到一个{@link RawDataSet}对象
 * @param <F> 指定生成的{@link RawDataSet}的特征变量的类型
 * @param <L> 指定生成的{@link RawDataSet}的标签变量的类型
 */
public class MakeLabelTransform<F,L> extends BaseTransform<ArrayList<F>, RawDataSet<F,L>> {

    private LabelMaker<F,L> labelMaker;

    public MakeLabelTransform(LabelMaker<F,L> labelMaker){
        this.labelMaker = labelMaker;
    }

    @SuppressWarnings("unchecked")
    @Override
    public Object tran2(ArrayList<F> fs) {
        return new RawDataSet(fs, labelMaker.makeAll(fs));
    }

    public void setLabelMaker(LabelMaker<F,L> labelMaker) {
        this.labelMaker = labelMaker;
    }

    public LabelMaker<F,L> getLabelMaker() {
        return labelMaker;
    }
}
