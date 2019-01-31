package com.jstarcraft.module.datatran;

import com.jstarcraft.module.datatran.transform.pretreatment.ListDisplayTransform;

import java.util.ArrayList;
/**
 * 抽象的数据集类 数据在被正式转换为nd4j的{@link org.nd4j.linalg.dataset.DataSet}类之前 可以选择以该类的对象为载体进行转换 所以称之为Raw
 * @param <F> 指定了数据集的特征存放在什么类型的变量中
 * @param <L> 指定了数据集的标签存放在什么类型的变量中
 */
public class RawDataSet<F, L> {
    /**
     * 数据集中的特征集合
     */
    private ArrayList<F> features;
    /**
     * 数据集中的标签集合
     */
    private ArrayList<L> labels;
    /**
     * 被使用于{@link RawDataSet#toString()}方法中
     */
    private ListDisplayTransform listDisplayTransform = new ListDisplayTransform(5, true);

    /**
     * 构造函数
     * @param features
     * @param labels
     */
    public RawDataSet(ArrayList<F> features, ArrayList<L> labels) {
        this.features = features;
        this.labels = labels;
    }

    /**
     *
     * @return
     */
    @Override
    public String toString() {
        return "===========INPUT===================\n"+listDisplayTransform.tran(features)
                +"===========OUTPUT==================\n"+listDisplayTransform.tran(labels);
    }

    /**
     * 返回{@link RawDataSet#features}对象的大小 同时也是数据集的数据条数
     * @return
     */
    public int size() {
        return features.size();
    }

    public ArrayList<F> getFeatures() {
        return features;
    }

    public ArrayList<L> getLabels() {
        return labels;
    }

    public void setFeatures(ArrayList<F> features) {
        this.features = features;
    }

    public void setLabels(ArrayList<L> labels) {
        this.labels = labels;
    }

    public ListDisplayTransform getListDisplayTransform() {
        return listDisplayTransform;
    }

    public void setListDisplayTransform(ListDisplayTransform listDisplayTransform) {
        this.listDisplayTransform = listDisplayTransform;
    }
}