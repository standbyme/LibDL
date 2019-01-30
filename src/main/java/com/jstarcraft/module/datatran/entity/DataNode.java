package com.jstarcraft.module.datatran.entity;

import com.jstarcraft.module.datatran.transform.MultiTransform;
import com.jstarcraft.module.datatran.transform.Transform;

public class DataNode<OUT> {
    private Object oriData;
    private MultiTransform<OUT> transform = new MultiTransform<>();

    public DataNode(Object oriData, Transform... transforms) {
        this.oriData = oriData;
        this.transform.add(transforms);
    }

    public void add(Transform... transforms) {
        this.transform.add(transforms);
    }

    public OUT translation() throws Exception{
        return transform.tran(oriData);
    }

    public Object getOriData() {
        return oriData;
    }

    public void setOriData(Object oriData) {
        this.oriData = oriData;
    }

    public MultiTransform<OUT> getTransform() {
        return transform;
    }

    public void setTransform(MultiTransform<OUT> transform) {
        this.transform = transform;
    }
}
