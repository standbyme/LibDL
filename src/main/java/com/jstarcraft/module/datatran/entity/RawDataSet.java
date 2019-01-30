package com.jstarcraft.module.datatran.entity;

import com.jstarcraft.module.datatran.transform.pretreatment.ListDisplayTransform;

import java.util.ArrayList;

public class RawDataSet<F, L> {
    private ArrayList<F> features;
    private ArrayList<L> labels;

    private ListDisplayTransform listDisplayTransform = new ListDisplayTransform(5, true);

    public RawDataSet(ArrayList<F> features, ArrayList<L> labels) {
        this.features = features;
        this.labels = labels;
    }

    @Override
    public String toString() {
        return "===========INPUT===================\n"+listDisplayTransform.tran(features)
                +"===========OUTPUT==================\n"+listDisplayTransform.tran(labels);
    }

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
